#!/usr/bin/env python3
"""
Example usage of the YOLOv11 Tiny Object Detection System
Demonstrates how to use the system for motor control applications
"""

import cv2
import numpy as np
from yolo_detection_system import YOLOv11TinyObjectDetector, MotorCoordinates
import argparse
import sys
from pathlib import Path

def create_test_image():
    """Create a test image with small objects for demonstration"""
    # Create a high-resolution test image (1920x1080)
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Add some small rectangular "objects" to simulate tiny targets
    objects = [
        (100, 100, 20, 15),   # x, y, width, height
        (500, 300, 15, 12),
        (800, 200, 25, 18),
        (1200, 400, 18, 14),
        (1500, 600, 22, 16),
        (300, 800, 16, 13),
    ]

    for x, y, w, h in objects:
        # Draw small colored rectangles
        color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
        # Add a border to make them more visible
        cv2.rectangle(image, (x-1, y-1), (x + w + 1, y + h + 1), (255, 255, 255), 1)

    return image

def process_single_image(detector, image_path, motor_config):
    """Process a single image and return motor coordinates"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return []

        print(f"Processing image: {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")

        # Process image
        motor_coordinates = detector.process_image(
            image,
            motor_x_range=motor_config['x_range'],
            motor_y_range=motor_config['y_range']
        )

        return motor_coordinates

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

def visualize_detections(image, motor_coordinates, coordinate_mapper):
    """Visualize detections on the image"""
    vis_image = image.copy()

    for i, coord in enumerate(motor_coordinates):
        # Convert motor coordinates back to image coordinates for visualization
        norm_x = (coord.x - coordinate_mapper.motor_x_range[0]) / (coordinate_mapper.motor_x_range[1] - coordinate_mapper.motor_x_range[0])
        norm_y = (coord.y - coordinate_mapper.motor_y_range[0]) / (coordinate_mapper.motor_y_range[1] - coordinate_mapper.motor_y_range[0])

        img_x = int(norm_x * coordinate_mapper.image_width)
        img_y = int(norm_y * coordinate_mapper.image_height)

        # Draw detection point
        cv2.circle(vis_image, (img_x, img_y), 10, (0, 255, 0), 2)
        cv2.putText(vis_image, f"{i+1}", (img_x + 15, img_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return vis_image

def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Tiny Object Detection for Motor Control")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="yolov11s.pt", help="Path to YOLOv11 model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--motor-x-min", type=float, default=-150.0, help="Motor X minimum position")
    parser.add_argument("--motor-x-max", type=float, default=150.0, help="Motor X maximum position")
    parser.add_argument("--motor-y-min", type=float, default=-150.0, help="Motor Y minimum position")
    parser.add_argument("--motor-y-max", type=float, default=150.0, help="Motor Y maximum position")
    parser.add_argument("--create-test", action="store_true", help="Create test image instead of using input")
    parser.add_argument("--visualize", action="store_true", help="Show visualization of detections")

    args = parser.parse_args()

    # Motor configuration
    motor_config = {
        'x_range': (args.motor_x_min, args.motor_x_max),
        'y_range': (args.motor_y_min, args.motor_y_max)
    }

    # Initialize detector
    print("Initializing YOLOv11 Tiny Object Detector...")
    try:
        detector = YOLOv11TinyObjectDetector(
            model_path=args.model,
            confidence_threshold=args.conf,
            iou_threshold=args.iou
        )
    except Exception as e:
        print(f"Error initializing detector: {e}")
        print("Make sure you have installed the requirements: pip install -r requirements.txt")
        return 1

    # Get image
    if args.create_test:
        print("Creating test image...")
        image = create_test_image()
        cv2.imwrite("test_image.jpg", image)
        image_path = "test_image.jpg"
        print("Test image saved as 'test_image.jpg'")
    elif args.image:
        image_path = args.image
        image = cv2.imread(image_path)
    else:
        print("Error: Please provide --image path or use --create-test")
        return 1

    # Process image
    motor_coordinates = process_single_image(detector, image_path, motor_config)

    # Display results
    print(f"\n{'='*60}")
    print(f"DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Total objects detected: {len(motor_coordinates)}")
    print()

    if motor_coordinates:
        print("Motor Coordinates:")
        print("-" * 60)
        print(f"{'#':<3} {'X':<10} {'Y':<10} {'Z':<10} {'Confidence':<12}")
        print("-" * 60)

        for i, coord in enumerate(motor_coordinates, 1):
            print(f"{i:<3} {coord.x:<10.2f} {coord.y:<10.2f} {coord.z:<10.2f} {coord.confidence:<12.3f}")

        print("\nMotor Control Commands:")
        print("-" * 40)
        for i, coord in enumerate(motor_coordinates, 1):
            print(f"Target {i}: MOVE_TO({coord.x:.2f}, {coord.y:.2f}, {coord.z:.2f})")
    else:
        print("No objects detected. Try:")
        print("- Lowering confidence threshold (--conf)")
        print("- Using a different model (--model)")
        print("- Checking image quality and lighting")

    # Visualization
    if args.visualize and motor_coordinates:
        try:
            from yolo_detection_system import MotorCoordinateMapper

            h, w = image.shape[:2]
            coordinate_mapper = MotorCoordinateMapper(w, h, motor_config['x_range'], motor_config['y_range'])

            vis_image = visualize_detections(image, motor_coordinates, coordinate_mapper)

            # Resize for display if too large
            max_display_size = 1200
            if max(h, w) > max_display_size:
                scale = max_display_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                vis_image = cv2.resize(vis_image, (new_w, new_h))

            cv2.imshow("Detections", vis_image)
            print("\nPress any key to close the visualization window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Visualization error: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
