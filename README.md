# YOLOv11 Tiny Object Detection System with SAHI

A high-precision object detection system optimized for tiny objects using YOLOv11 with P2 head and SAHI (Slicing Aided Hyper Inference). Designed to output motor control coordinates for robotic applications.

## Features

- **YOLOv11-s/n with P2 Head**: Enhanced detection head with stride-4 feature maps for tiny object detection
- **SAHI Integration**: Sliding window inference with configurable overlap for high-resolution images
- **Motor Coordinate Mapping**: Automatic conversion from image coordinates to motor control positions
- **High-Resolution Support**: Optimized for 1280-1920px images with motion blur reduction
- **Configurable Parameters**: Adjustable confidence thresholds, IoU settings, and motor ranges

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. The system will automatically download YOLOv11 models on first use.

## Quick Start

### Basic Usage

```python
from yolo_detection_system import YOLOv11TinyObjectDetector
import cv2

# Initialize detector
detector = YOLOv11TinyObjectDetector(
    model_path="yolov11s.pt",
    confidence_threshold=0.25,
    iou_threshold=0.45
)

# Load image
image = cv2.imread("your_image.jpg")

# Process and get motor coordinates
motor_coordinates = detector.process_image(
    image,
    motor_x_range=(-150.0, 150.0),
    motor_y_range=(-150.0, 150.0)
)

# Use coordinates for motor control
for coord in motor_coordinates:
    print(f"Move to: X={coord.x:.2f}, Y={coord.y:.2f}, Z={coord.z:.2f}")
    print(f"Confidence: {coord.confidence:.3f}")
```

### Command Line Usage

```bash
# Process an image
python example_usage.py --image path/to/image.jpg --visualize

# Create and process test image
python example_usage.py --create-test --visualize

# Custom motor ranges
python example_usage.py --image image.jpg --motor-x-min -200 --motor-x-max 200

# Adjust detection sensitivity
python example_usage.py --image image.jpg --conf 0.15 --iou 0.5
```

## System Architecture

### 1. P2 Detection Head
- Adds stride-4 feature maps for enhanced tiny object detection
- Improves recall on objects smaller than 32x32 pixels
- Custom anchor configuration optimized for small targets

### 2. SAHI Processing
- Configurable slice size (default: 640x640)
- Overlap ratios: 0.2-0.3 for height/width
- NMS-based detection merging to eliminate duplicates
- Significant recall improvement for small objects

### 3. Motor Coordinate Mapping
- Linear mapping from image coordinates to motor positions
- Configurable motor ranges for X, Y, Z axes
- Maintains detection confidence scores
- Supports various coordinate systems

## Configuration Options

### Detection Parameters
- `confidence_threshold`: Minimum detection confidence (default: 0.25)
- `iou_threshold`: IoU threshold for NMS (default: 0.45)
- `model_path`: YOLOv11 model variant (yolov11n.pt, yolov11s.pt, etc.)

### SAHI Parameters
- `slice_height/width`: Window size for sliding inference (default: 640)
- `overlap_height/width_ratio`: Overlap between windows (default: 0.2)

### Motor Parameters
- `motor_x_range`: X-axis motor position limits
- `motor_y_range`: Y-axis motor position limits
- `motor_z_default`: Default Z-axis position

## Optimization Tips

### For Maximum Tiny Object Detection:
1. Use high-resolution input images (1280-1920px)
2. Reduce confidence threshold to 0.15-0.2
3. Increase SAHI overlap to 0.3
4. Use YOLOv11n for speed or YOLOv11s for accuracy

### For Speed Optimization:
1. Reduce input image size to 1280px
2. Increase slice size to 800x800
3. Reduce overlap to 0.1
4. Use YOLOv11n model

### For Motion Blur Reduction:
1. Use short exposure times
2. High shutter speeds
3. Proper lighting to avoid high ISO
4. Consider image stabilization

## Example Applications

- **PCB Component Inspection**: Detect tiny components on circuit boards
- **Quality Control**: Identify small defects in manufacturing
- **Microscopy Analysis**: Locate cellular structures or particles
- **Drone Surveillance**: Track small objects from aerial views
- **Robotic Pick-and-Place**: Precise positioning for small parts

## API Reference

### YOLOv11TinyObjectDetector

Main detection class combining YOLOv11, P2 head, and SAHI.

#### Methods:
- `process_image(image, motor_x_range, motor_y_range)`: Main processing pipeline
- `preprocess_image(image, target_size)`: Image preprocessing for high-resolution
- `detect_objects(image_slice)`: Run detection on single slice

### MotorCoordinates

Data class for motor control output.

#### Attributes:
- `x, y, z`: Motor position coordinates
- `confidence`: Detection confidence score

### SAHIProcessor

Handles sliding window inference.

#### Methods:
- `get_slices(image_height, image_width)`: Calculate slice coordinates
- `merge_detections(detections, slices, iou_threshold)`: Merge overlapping detections

## Troubleshooting

### Common Issues:

1. **No detections found**:
   - Lower confidence threshold
   - Check image quality and lighting
   - Verify object size (should be >10 pixels)

2. **Too many false positives**:
   - Increase confidence threshold
   - Adjust IoU threshold
   - Use larger model variant

3. **Slow processing**:
   - Reduce input image size
   - Increase slice size
   - Reduce overlap ratio
   - Use GPU acceleration

4. **Memory errors**:
   - Reduce slice size
   - Process images sequentially
   - Use smaller model variant

## Performance Benchmarks

Typical performance on 1920x1080 images:
- **YOLOv11n + SAHI**: ~2-3 FPS, high recall on tiny objects
- **YOLOv11s + SAHI**: ~1-2 FPS, maximum accuracy
- **Memory usage**: 2-4GB GPU memory depending on slice configuration

## License

This project uses the Ultralytics YOLOv11 framework. Please refer to their license terms for commercial usage.
