import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
import threading
from datetime import datetime
import math

class DistanceTracker:
    def __init__(self, demo_mode=False):
        self.demo_mode = demo_mode
        
        # Camera parameters (MacBook webcam typical values)
        # These would ideally come from camera calibration
        self.camera_fov_horizontal = 78  # degrees (typical for MacBook)
        self.camera_fov_vertical = 58    # degrees
        self.camera_height = 0.3         # meters above table/ground
        self.camera_tilt = 10            # degrees downward
        
        # Frame dimensions (will be updated when we get actual frames)
        self.frame_width = 640
        self.frame_height = 480
        
        # Initialize video capture
        if not demo_mode:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("Camera not accessible. Switching to demo mode...")
                    self.demo_mode = True
                else:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    print("Webcam initialized successfully!")
            except Exception as e:
                print(f"Webcam error: {e}. Switching to demo mode...")
                self.demo_mode = True
        
        if self.demo_mode:
            from pathlib import Path
            import random
            self.frame_dir = Path("/Users/canis/dev/argoone-labs/frames")
            self.available_frames = list(self.frame_dir.glob("frame_*.png"))
            if len(self.available_frames) < 10:
                raise RuntimeError("Demo mode requires at least 10 frame files")
            print(f"Demo mode: Using {len(self.available_frames)} frames")
        
        # Frame storage
        self.previous_frame = None
        self.frame_count = 0
        
        # Display data
        self.current_frame = None
        self.difference_image = None
        self.detected_objects = []
        self.detection_info = "Starting..."
        
        # Control flags
        self.running = True
        self.detection_thread = None
        
        # Setup matplotlib
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 9))
        title = 'Distance Tracker - Demo' if self.demo_mode else 'Distance Tracker - Live'
        self.fig.suptitle(title, fontsize=14)
        
        # Initialize subplots
        self.axes[0, 0].set_title('Current Frame')
        self.axes[0, 1].set_title('Frame Difference')
        self.axes[1, 0].set_title('Detected Objects with Distances')
        self.axes[1, 1].set_title('Top-Down View (Estimated Positions)')
        
        for ax in self.axes.flat:
            ax.axis('off')
        
        # Status text
        self.status_text = self.fig.text(0.5, 0.02, 'Starting...', 
                                        ha='center', fontsize=10)
        
        print(f"📐 Camera Setup:")
        print(f"  FOV: {self.camera_fov_horizontal}° × {self.camera_fov_vertical}°")
        print(f"  Height: {self.camera_height}m, Tilt: {self.camera_tilt}°")
        print(f"  Frame: {self.frame_width}×{self.frame_height}")
        
        # Start processing
        self.start_detection_thread()
    
    def capture_frame(self):
        """Capture frame from webcam or demo files"""
        if self.demo_mode:
            import random
            frame_path = random.choice(self.available_frames)
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                frame = cv2.flip(frame, 1)  # Horizontal flip
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        else:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Horizontal flip
                self.frame_height, self.frame_width = frame.shape[:2]
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
    
    def pixel_to_world_coords(self, pixel_x, pixel_y):
        """Convert pixel coordinates to real-world coordinates"""
        # Convert pixel coordinates to normalized coordinates (-1 to 1)
        norm_x = (pixel_x - self.frame_width / 2) / (self.frame_width / 2)
        norm_y = (pixel_y - self.frame_height / 2) / (self.frame_height / 2)
        
        # Calculate horizontal and vertical angles from camera center
        angle_horizontal = norm_x * (self.camera_fov_horizontal / 2) * math.pi / 180
        angle_vertical = norm_y * (self.camera_fov_vertical / 2) * math.pi / 180
        
        # Adjust vertical angle for camera tilt
        angle_vertical_adjusted = angle_vertical - (self.camera_tilt * math.pi / 180)
        
        # Calculate distance to ground plane intersection
        if abs(angle_vertical_adjusted) < 0.01:  # Avoid division by zero
            distance_to_point = 50  # Default large distance
        else:
            # Distance to where the ray hits the ground plane
            distance_to_point = self.camera_height / math.tan(abs(angle_vertical_adjusted))
        
        # Limit reasonable distance range
        distance_to_point = max(0.1, min(distance_to_point, 10.0))
        
        # Calculate real-world x, z coordinates (camera-relative)
        world_x = distance_to_point * math.tan(angle_horizontal)
        world_z = distance_to_point
        
        return world_x, world_z, distance_to_point
    
    def estimate_object_size(self, bbox_width, bbox_height, distance):
        """Estimate real-world object size based on pixel size and distance"""
        # Calculate angular size of the bounding box
        angular_width = (bbox_width / self.frame_width) * self.camera_fov_horizontal * math.pi / 180
        angular_height = (bbox_height / self.frame_height) * self.camera_fov_vertical * math.pi / 180
        
        # Convert to real-world size
        real_width = 2 * distance * math.tan(angular_width / 2)
        real_height = 2 * distance * math.tan(angular_height / 2)
        
        return real_width, real_height
    
    def simple_frame_difference(self, frame1, frame2):
        """Simple frame difference detection"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Threshold
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 500:  # Minimum area
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w < 15 or h < 15 or w > 300 or h > 300:
                continue
            
            # Calculate center point
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Convert to world coordinates
            world_x, world_z, distance = self.pixel_to_world_coords(center_x, center_y)
            
            # Estimate object size
            real_width, real_height = self.estimate_object_size(w, h, distance)
            
            objects.append({
                'bbox': (x, y, w, h),
                'center_pixel': (center_x, center_y),
                'world_coords': (world_x, world_z),
                'distance': distance,
                'size_real': (real_width, real_height),
                'area': area
            })
        
        # Sort by area
        objects.sort(key=lambda x: x['area'], reverse=True)
        return thresh, objects[:5]  # Max 5 objects
    
    def detection_loop(self):
        """Main detection loop"""
        while self.running:
            try:
                # Capture frame
                frame_rgb = self.capture_frame()
                if frame_rgb is None:
                    time.sleep(0.1)
                    continue
                
                self.current_frame = frame_rgb
                self.frame_count += 1
                
                # Skip first frame
                if self.previous_frame is None:
                    self.previous_frame = frame_rgb
                    time.sleep(0.1)
                    continue
                
                # Detect objects using simple frame difference
                diff_image, objects = self.simple_frame_difference(self.previous_frame, frame_rgb)
                
                self.difference_image = diff_image
                self.detected_objects = objects
                
                # Update status
                num_objects = len(objects)
                self.detection_info = f"{num_objects} object{'s' if num_objects != 1 else ''} detected"
                
                # Print status
                if self.frame_count % 30 == 0 or num_objects > 0:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\n[{timestamp}] Frame {self.frame_count}: {self.detection_info}")
                    
                    for i, obj in enumerate(objects):
                        x, y, w, h = obj['bbox']
                        world_x, world_z = obj['world_coords']
                        distance = obj['distance']
                        real_w, real_h = obj['size_real']
                        
                        print(f"  Object {i+1}:")
                        print(f"    Pixel: ({x},{y}) {w}×{h}px")
                        print(f"    World: ({world_x:.2f}m, {world_z:.2f}m)")
                        print(f"    Distance: {distance:.2f}m")
                        print(f"    Real size: {real_w:.2f}m × {real_h:.2f}m")
                
                # Update previous frame
                self.previous_frame = frame_rgb
                
                # Control frame rate
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                print(f"Detection error: {e}")
                time.sleep(0.1)
    
    def start_detection_thread(self):
        """Start the detection thread"""
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
    
    def update_plot(self, frame_num):
        """Update the matplotlib visualization"""
        try:
            # Clear all axes
            for ax in self.axes.flat:
                ax.clear()
            
            # Current frame
            if self.current_frame is not None:
                self.axes[0, 0].imshow(self.current_frame)
                self.axes[0, 0].set_title('Current Frame')
                self.axes[0, 0].axis('off')
            
            # Frame difference
            if self.difference_image is not None:
                self.axes[0, 1].imshow(self.difference_image, cmap='gray')
                self.axes[0, 1].set_title('Frame Difference')
                self.axes[0, 1].axis('off')
            
            # Detected objects with distance info
            if self.current_frame is not None:
                self.axes[1, 0].imshow(self.current_frame)
                
                # Draw bounding boxes and distance info
                colors = ['red', 'blue', 'green', 'yellow', 'cyan']
                for i, obj in enumerate(self.detected_objects):
                    x, y, w, h = obj['bbox']
                    distance = obj['distance']
                    real_w, real_h = obj['size_real']
                    
                    color = colors[i % len(colors)]
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                           edgecolor=color, facecolor='none')
                    self.axes[1, 0].add_patch(rect)
                    
                    # Distance label
                    label = f'{i+1}: {distance:.1f}m\n{real_w:.2f}×{real_h:.2f}m'
                    self.axes[1, 0].text(x, y-10, label, 
                                       color=color, fontweight='bold', fontsize=8,
                                       bbox=dict(boxstyle='round,pad=0.3', 
                                               facecolor='white', alpha=0.8))
                
                self.axes[1, 0].set_title('Detected Objects with Distances')
                self.axes[1, 0].axis('off')
            
            # Top-down view
            self.axes[1, 1].set_xlim(-3, 3)
            self.axes[1, 1].set_ylim(0, 5)
            self.axes[1, 1].set_aspect('equal')
            
            # Draw camera position
            self.axes[1, 1].plot(0, 0, 'ko', markersize=8, label='Camera')
            self.axes[1, 1].arrow(0, 0, 0, 0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
            
            # Draw detected objects in top-down view
            colors = ['red', 'blue', 'green', 'yellow', 'cyan']
            for i, obj in enumerate(self.detected_objects):
                world_x, world_z = obj['world_coords']
                color = colors[i % len(colors)]
                
                self.axes[1, 1].plot(world_x, world_z, 'o', color=color, markersize=10)
                self.axes[1, 1].text(world_x + 0.1, world_z + 0.1, f'{i+1}', 
                                   color=color, fontweight='bold', fontsize=10)
            
            self.axes[1, 1].grid(True, alpha=0.3)
            self.axes[1, 1].set_xlabel('Left/Right (m)')
            self.axes[1, 1].set_ylabel('Forward Distance (m)')
            self.axes[1, 1].set_title('Top-Down View (Estimated Positions)')
            
            # Update status
            mode_info = " (Demo)" if self.demo_mode else ""
            timestamp = datetime.now().strftime("%H:%M:%S")
            status_msg = f"{timestamp} | Frame {self.frame_count} | {self.detection_info}{mode_info}"
            self.status_text.set_text(status_msg)
        
        except Exception as e:
            print(f"Plot update error: {e}")
    
    def start_gui(self):
        """Start the GUI with animation"""
        try:
            self.animation = FuncAnimation(self.fig, self.update_plot, 
                                         interval=100, blit=False, cache_frame_data=False)
            
            plt.tight_layout()
            plt.show()
        
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop tracking and release resources"""
        print("Shutting down...")
        self.running = False
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        if hasattr(self, 'animation') and self.animation is not None:
            self.animation.event_source.stop()
        
        plt.close('all')

def main():
    print("📐 Distance Tracker")
    print("Simple frame difference + Real-world coordinate mapping")
    print("Estimates object distances and positions relative to camera")
    print()
    print("🔧 Camera calibration tips:")
    print("- Adjust camera_height in code to match your setup")
    print("- Measure actual distances to verify accuracy")
    print("- Works best for objects on a flat surface (table/floor)")
    print()
    
    try:
        tracker = DistanceTracker(demo_mode=False)
        tracker.start_gui()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if '--demo' in sys.argv:
        print("🎬 Demo Mode")
        tracker = DistanceTracker(demo_mode=True)
        tracker.start_gui()
    else:
        main() 