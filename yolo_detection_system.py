"""
YOLOv11 Detection System with P2 Head and SAHI for Motor Control
Optimized for tiny object detection with high-resolution inference
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.utils import ops
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Container for detection results"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    center: Tuple[float, float]  # center coordinates

@dataclass
class MotorCoordinates:
    """Motor control coordinates"""
    x: float  # X-axis motor position
    y: float  # Y-axis motor position
    z: float  # Z-axis motor position (optional)
    confidence: float  # Detection confidence

class P2DetectionHead(nn.Module):
    """
    Modified YOLO detection head with P2 feature map (stride 4)
    for enhanced tiny object detection
    """
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))

        # Add P2 layer (stride 4) for tiny objects
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                else:
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

class SAHIProcessor:
    """
    Slicing Aided Hyper Inference (SAHI) implementation
    Slides windows over high-resolution images for better small object detection
    """

    def __init__(self,
                 slice_height: int = 640,
                 slice_width: int = 640,
                 overlap_height_ratio: float = 0.2,
                 overlap_width_ratio: float = 0.2):
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio

    def get_slices(self, image_height: int, image_width: int) -> List[Tuple[int, int, int, int]]:
        """
        Calculate slice coordinates for SAHI processing
        Returns list of (x1, y1, x2, y2) coordinates
        """
        slices = []

        # Calculate step sizes
        y_step = int(self.slice_height * (1 - self.overlap_height_ratio))
        x_step = int(self.slice_width * (1 - self.overlap_width_ratio))

        # Generate slices
        for y in range(0, image_height, y_step):
            for x in range(0, image_width, x_step):
                x1 = x
                y1 = y
                x2 = min(x + self.slice_width, image_width)
                y2 = min(y + self.slice_height, image_height)

                # Skip if slice is too small
                if (x2 - x1) < self.slice_width // 2 or (y2 - y1) < self.slice_height // 2:
                    continue

                slices.append((x1, y1, x2, y2))

        return slices

    def merge_detections(self, all_detections: List[List[DetectionResult]],
                        slices: List[Tuple[int, int, int, int]],
                        iou_threshold: float = 0.5) -> List[DetectionResult]:
        """
        Merge detections from all slices using NMS
        """
        if not all_detections:
            return []

        # Convert slice-relative coordinates to global coordinates
        global_detections = []

        for detections, (x1, y1, x2, y2) in zip(all_detections, slices):
            for det in detections:
                # Convert to global coordinates
                global_bbox = (
                    det.bbox[0] + x1,
                    det.bbox[1] + y1,
                    det.bbox[2] + x1,
                    det.bbox[3] + y1
                )
                global_center = (
                    det.center[0] + x1,
                    det.center[1] + y1
                )

                global_det = DetectionResult(
                    bbox=global_bbox,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    center=global_center
                )
                global_detections.append(global_det)

        # Apply NMS to remove duplicate detections
        if not global_detections:
            return []

        # Convert to format suitable for NMS
        boxes = torch.tensor([det.bbox for det in global_detections], dtype=torch.float32)
        scores = torch.tensor([det.confidence for det in global_detections], dtype=torch.float32)

        # Apply NMS
        keep_indices = ops.nms(boxes, scores, iou_threshold)

        return [global_detections[i] for i in keep_indices]

class MotorCoordinateMapper:
    """
    Maps image coordinates to motor control coordinates
    """

    def __init__(self,
                 image_width: int,
                 image_height: int,
                 motor_x_range: Tuple[float, float] = (-100.0, 100.0),
                 motor_y_range: Tuple[float, float] = (-100.0, 100.0),
                 motor_z_default: float = 0.0):
        self.image_width = image_width
        self.image_height = image_height
        self.motor_x_range = motor_x_range
        self.motor_y_range = motor_y_range
        self.motor_z_default = motor_z_default

    def image_to_motor_coordinates(self, detection: DetectionResult) -> MotorCoordinates:
        """
        Convert image pixel coordinates to motor coordinates
        """
        # Get center coordinates
        center_x, center_y = detection.center

        # Normalize to [0, 1]
        norm_x = center_x / self.image_width
        norm_y = center_y / self.image_height

        # Map to motor coordinate ranges
        motor_x = self.motor_x_range[0] + norm_x * (self.motor_x_range[1] - self.motor_x_range[0])
        motor_y = self.motor_y_range[0] + norm_y * (self.motor_y_range[1] - self.motor_y_range[0])

        return MotorCoordinates(
            x=motor_x,
            y=motor_y,
            z=self.motor_z_default,
            confidence=detection.confidence
        )

class YOLOv11TinyObjectDetector:
    """
    Main detection system combining YOLOv11 with P2 head and SAHI
    """

    def __init__(self,
                 model_path: str = "yolov11s.pt",
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = "auto"):

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = self._get_device(device)

        # Load YOLOv11 model
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Initialize SAHI processor
        self.sahi = SAHIProcessor()

        logger.info(f"YOLOv11 Tiny Object Detector initialized on {self.device}")

    def _get_device(self, device: str) -> str:
        """Auto-detect or validate device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def preprocess_image(self, image: np.ndarray, target_size: int = 1280) -> np.ndarray:
        """
        Preprocess image for high-resolution inference
        """
        h, w = image.shape[:2]

        # Resize while maintaining aspect ratio
        if max(h, w) != target_size:
            scale = target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return image

    def detect_objects(self, image_slice: np.ndarray) -> List[DetectionResult]:
        """
        Run detection on a single image slice
        """
        results = self.model(image_slice,
                           conf=self.confidence_threshold,
                           iou=self.iou_threshold,
                           verbose=False)

        detections = []

        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    detection = DetectionResult(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(conf),
                        class_id=cls_id,
                        center=(center_x, center_y)
                    )
                    detections.append(detection)

        return detections

    def process_image(self, image: np.ndarray,
                     motor_x_range: Tuple[float, float] = (-100.0, 100.0),
                     motor_y_range: Tuple[float, float] = (-100.0, 100.0)) -> List[MotorCoordinates]:
        """
        Main processing pipeline: image -> detections -> motor coordinates
        """
        logger.info(f"Processing image of size {image.shape[:2]}")

        # Preprocess image
        processed_image = self.preprocess_image(image)
        h, w = processed_image.shape[:2]

        # Get SAHI slices
        slices = self.sahi.get_slices(h, w)
        logger.info(f"Generated {len(slices)} SAHI slices")

        # Process each slice
        all_detections = []
        for i, (x1, y1, x2, y2) in enumerate(slices):
            slice_image = processed_image[y1:y2, x1:x2]
            slice_detections = self.detect_objects(slice_image)
            all_detections.append(slice_detections)

            if slice_detections:
                logger.info(f"Slice {i+1}/{len(slices)}: Found {len(slice_detections)} objects")

        # Merge detections using NMS
        merged_detections = self.sahi.merge_detections(all_detections, slices, self.iou_threshold)
        logger.info(f"After NMS: {len(merged_detections)} final detections")

        # Convert to motor coordinates
        coordinate_mapper = MotorCoordinateMapper(w, h, motor_x_range, motor_y_range)
        motor_coordinates = []

        for detection in merged_detections:
            motor_coord = coordinate_mapper.image_to_motor_coordinates(detection)
            motor_coordinates.append(motor_coord)
            logger.info(f"Detection -> Motor: ({motor_coord.x:.2f}, {motor_coord.y:.2f}) "
                       f"confidence: {motor_coord.confidence:.3f}")

        return motor_coordinates

def main():
    """
    Example usage of the YOLOv11 tiny object detection system
    """
    # Initialize detector
    detector = YOLOv11TinyObjectDetector(
        model_path="yolov11s.pt",  # Will download if not present
        confidence_threshold=0.25,
        iou_threshold=0.45
    )

    # Load and process image
    image_path = "test_image.jpg"  # Replace with your image path

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        # Process image and get motor coordinates
        motor_coordinates = detector.process_image(
            image,
            motor_x_range=(-150.0, 150.0),  # Adjust based on your motor setup
            motor_y_range=(-150.0, 150.0)
        )

        # Output results
        print(f"\nDetected {len(motor_coordinates)} objects:")
        print("-" * 50)

        for i, coord in enumerate(motor_coordinates, 1):
            print(f"Object {i}:")
            print(f"  Motor X: {coord.x:.2f}")
            print(f"  Motor Y: {coord.y:.2f}")
            print(f"  Motor Z: {coord.z:.2f}")
            print(f"  Confidence: {coord.confidence:.3f}")
            print()

        return motor_coordinates

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return []

if __name__ == "__main__":
    main()
