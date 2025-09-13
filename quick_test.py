#!/usr/bin/env python3
"""
Quick test script for the YOLOv11 detection system
Creates a test image and demonstrates the complete pipeline
"""

import cv2
import numpy as np
from yolo_detection_system import YOLOv11TinyObjectDetector

def create_test_image_with_objects():
    """Create a test image with small objects"""
    # Create 1280x720 test image
    image = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Add background pattern
    image[:] = (30, 30, 30)

    # Add small test objects at various locations
    test_objects = [
        (100, 100, 25, 20, (255, 100, 100)),  # x, y, w, h, color
        (300, 200, 20, 15, (100, 255, 100)),
        (500, 150, 30, 25, (100, 100, 255)),
        (800, 300, 18, 22, (255, 255, 100)),
        (1000, 400, 22, 18, (255, 100, 255)),
        (200, 500, 16, 20, (100, 255, 255)),
    ]

    for x, y, w, h, color in test_objects:
        # Draw filled rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
        # Add white border to make it more visible
        cv2.rectangle(image, (x-1, y-1), (x + w + 1, y + h + 1), (255, 255, 255), 1)
        # Add center point
        center_x, center_y = x + w//2, y + h//2
        cv2.circle(image, (center_x, center_y), 2, (255, 255, 255), -1)

    return image

def main():
    print("YOLOv11 Tiny Object Detection - Quick Test")
    print("=" * 50)

    # Create test image
    print("Creating test image...")
    test_image = create_test_image_with_objects()
    cv2.imwrite("quick_test_image.jpg", test_image)
    print("Test image saved as 'quick_test_image.jpg'")

    try:
        # Initialize detector
        print("\nInitializing YOLOv11 detector...")
        detector = YOLOv11TinyObjectDetector(
            model_path="yolov11n.pt",  # Use nano version for faster download/processing
            confidence_threshold=0.1,   # Lower threshold for test objects
            iou_threshold=0.45
        )

        # Process test image
        print("Processing test image...")
        motor_coordinates = detector.process_image(
            test_image,
            motor_x_range=(-100.0, 100.0),
            motor_y_range=(-100.0, 100.0)
        )

        # Display results
        print(f"\nResults:")
        print(f"Detected {len(motor_coordinates)} objects")

        if motor_coordinates:
            print("\nMotor Coordinates:")
            for i, coord in enumerate(motor_coordinates, 1):
                print(f"  Object {i}: X={coord.x:6.2f}, Y={coord.y:6.2f}, Conf={coord.confidence:.3f}")
        else:
            print("No objects detected. This might be normal for the test image.")
            print("Try running with a real image containing recognizable objects.")

        print(f"\nTest completed successfully!")
        print(f"You can now use the system with real images.")

    except Exception as e:
        print(f"Error during test: {e}")
        print("\nPossible solutions:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Ensure you have internet connection (for model download)")
        print("3. Check if you have sufficient disk space")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
