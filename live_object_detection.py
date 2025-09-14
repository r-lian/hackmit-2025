import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
import threading
from datetime import datetime
import os
from pathlib import Path
import random

class LiveObjectDetector:
    def __init__(self, demo_mode=False):
        self.demo_mode = demo_mode
        
        if not demo_mode:
            # Try to initialize webcam
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("Camera not accessible. Switching to demo mode...")
                    self.demo_mode = True
                else:
                    # Set webcam properties for better performance
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    print("Webcam initialized successfully!")
            except Exception as e:
                print(f"Webcam error: {e}. Switching to demo mode...")
                self.demo_mode = True
        
        if self.demo_mode:
            # Setup demo mode with existing frames
            self.frame_dir = Path("/Users/canis/dev/argoone-labs/frames")
            self.available_frames = list(self.frame_dir.glob("frame_*.png"))
            if len(self.available_frames) < 2:
                raise RuntimeError("Demo mode requires at least 2 frame files in the frames directory")
            print(f"Demo mode: Using {len(self.available_frames)} frames from frames directory")
        
        # Initialize frames and detection results
        self.frame1 = None
        self.frame2 = None
        self.diff_image = None
        self.bounding_box = None
        self.detection_info = ""
        self.last_detection_time = ""
        
        # Control flags
        self.running = True
        self.detection_thread = None
        
        # Setup matplotlib figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        title = 'Live Moving Object Detection - Demo Mode' if self.demo_mode else 'Live Moving Object Detection - MacBook Webcam'
        self.fig.suptitle(title, fontsize=16)
        
        # Initialize subplot titles
        self.axes[0, 0].set_title('Frame 1 (t=0)')
        self.axes[0, 1].set_title('Frame 2 (t=+1s)')
        self.axes[1, 0].set_title('Thresholded Difference')
        self.axes[1, 1].set_title('Detection Result')
        
        # Turn off axes for all subplots
        for ax in self.axes.flat:
            ax.axis('off')
        
        # Add status text
        status_msg = 'Demo mode active - using sample frames' if self.demo_mode else 'Starting webcam...'
        self.status_text = self.fig.text(0.5, 0.02, status_msg, 
                                        ha='center', fontsize=10)
        
        # Start detection thread
        self.start_detection_thread()
    
    def capture_frame(self):
        """Capture a frame from webcam or load from demo files"""
        if self.demo_mode:
            # Pick a random frame from the available frames
            frame_path = random.choice(self.available_frames)
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        else:
            # Capture from webcam
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
    
    def detect_moving_object(self, frame1, frame2):
        """Detect moving object between two frames"""
        if frame1 is None or frame2 is None:
            return None, None, ""
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Apply threshold - lower threshold for demo mode since frames might be more similar
        threshold_value = 15
        _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return thresh, None, "No moving objects detected"
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        
        # Filter out very small movements (noise) - lower threshold for demo mode
        min_area = 50 if self.demo_mode else 100
        if contour_area < min_area:
            return thresh, None, f"Movement too small (area: {contour_area:.1f})"
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        info = f"Object detected - Area: {contour_area:.1f}px, Box: ({x},{y}) {w}x{h}"
        
        return thresh, (x, y, w, h), info
    
    def detection_loop(self):
        """Main detection loop that runs every 5 seconds"""
        cycle_count = 0
        while self.running:
            try:
                cycle_count += 1
                mode_info = f"[Demo cycle {cycle_count}]" if self.demo_mode else ""
                
                # Capture first frame
                print(f"Capturing frame 1... {mode_info}")
                frame1 = self.capture_frame()
                
                if frame1 is not None:
                    # Wait 1 second
                    time.sleep(1/240)
                    
                    # Capture second frame
                    print(f"Capturing frame 2... {mode_info}")
                    frame2 = self.capture_frame()
                    
                    if frame2 is not None:
                        # Detect moving object
                        diff_img, bbox, info = self.detect_moving_object(frame1, frame2)
                        
                        # Update stored results
                        self.frame1 = frame1
                        self.frame2 = frame2
                        self.diff_image = diff_img
                        self.bounding_box = bbox
                        self.detection_info = info
                        self.last_detection_time = datetime.now().strftime("%H:%M:%S")
                        
                        print(f"[{self.last_detection_time}] {info} {mode_info}")
                
                # Wait for the remainder of 5 seconds
                time.sleep(1/240)
                
            except Exception as e:
                print(f"Detection error: {e}")
                time.sleep(1.0)
    
    def start_detection_thread(self):
        """Start the detection thread"""
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
    
    def update_plot(self, frame_num):
        """Update the matplotlib plots"""
        try:
            # Clear all axes
            for ax in self.axes.flat:
                ax.clear()
                ax.axis('off')
            
            if self.frame1 is not None:
                # Plot frame 1
                self.axes[0, 0].imshow(self.frame1)
                self.axes[0, 0].set_title('Frame 1 (t=0)')
            
            if self.frame2 is not None:
                # Plot frame 2
                self.axes[0, 1].imshow(self.frame2)
                self.axes[0, 1].set_title('Frame 2 (t=+1s)')
            
            if self.diff_image is not None:
                # Plot difference
                self.axes[1, 0].imshow(self.diff_image, cmap='gray')
                self.axes[1, 0].set_title('Thresholded Difference')
            
            if self.frame2 is not None:
                # Plot frame 2 with bounding box
                self.axes[1, 1].imshow(self.frame2)
                
                if self.bounding_box is not None:
                    x, y, w, h = self.bounding_box
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                           edgecolor='red', facecolor='none')
                    self.axes[1, 1].add_patch(rect)
                    
                    # Add bounding box text
                    self.axes[1, 1].text(x, y-10, f'({x},{y}) {w}x{h}', 
                                       color='red', fontweight='bold', fontsize=8,
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                self.axes[1, 1].set_title('Detection Result')
            
            # Update status text
            mode_info = " (Demo Mode)" if self.demo_mode else ""
            status_msg = f"Last detection: {self.last_detection_time} | {self.detection_info}{mode_info}"
            self.status_text.set_text(status_msg)
            
            # Turn off axes for all subplots
            for ax in self.axes.flat:
                ax.axis('off')
        
        except Exception as e:
            print(f"Plot update error: {e}")
    
    def start_gui(self):
        """Start the GUI with animation"""
        try:
            # Create animation that updates every 100ms
            self.animation = FuncAnimation(self.fig, self.update_plot, 
                                         interval=100, blit=False, cache_frame_data=False)
            
            # Show the plot
            plt.tight_layout()
            plt.show()
            
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the detection and release resources"""
        print("Shutting down...")
        self.running = False
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        if hasattr(self, 'animation') and self.animation is not None:
            self.animation.event_source.stop()
        
        plt.close('all')

def main():
    print("Starting Live Object Detection...")
    print("Press Ctrl+C or close the window to stop")
    print()
    
    try:
        # First try normal webcam mode
        detector = LiveObjectDetector(demo_mode=False)
        detector.start_gui()
    except Exception as e:
        print(f"Error: {e}")
        print("\nCamera Access Issues:")
        print("1. Go to System Preferences (or System Settings)")
        print("2. Navigate to Security & Privacy → Camera")
        print("3. Allow Terminal or your Python app to access the camera")
        print("4. Restart the terminal and try again")
        print("\nAlternatively, you can run in demo mode:")
        print("python live_object_detection.py --demo")

if __name__ == "__main__":
    detector = LiveObjectDetector(demo_mode=False)
    detector.start_gui()