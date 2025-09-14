import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
import threading
from datetime import datetime

class RefinedTracker:
    def __init__(self, demo_mode=False):
        self.demo_mode = demo_mode
        
        # Initialize video capture
        if not demo_mode:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("Camera not accessible. Switching to demo mode...")
                    self.demo_mode = True
                else:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
        
        # Feature detector for jitter correction
        self.orb = cv2.ORB_create(nfeatures=200)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Frame storage
        self.previous_frame = None
        self.frame_count = 0
        
        # Display data
        self.current_frame = None
        self.corrected_frame = None
        self.difference_image = None
        self.detected_objects = []
        self.detection_info = "Starting..."
        self.jitter_info = ""
        
        # Control flags
        self.running = True
        self.detection_thread = None
        
        # Setup matplotlib
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        title = 'Refined Tracker - Demo' if self.demo_mode else 'Refined Tracker - Live'
        self.fig.suptitle(title, fontsize=14)
        
        # Initialize subplots
        self.axes[0, 0].set_title('Current Frame')
        self.axes[0, 1].set_title('Jitter-Corrected Frame')
        self.axes[1, 0].set_title('Frame Difference (After Correction)')
        self.axes[1, 1].set_title('Detected Objects')
        
        for ax in self.axes.flat:
            ax.axis('off')
        
        # Status text
        self.status_text = self.fig.text(0.5, 0.02, 'Starting...', 
                                        ha='center', fontsize=10)
        
        # Start processing
        self.start_detection_thread()
    
    def capture_frame(self):
        """Capture frame from webcam or demo files"""
        if self.demo_mode:
            import random
            frame_path = random.choice(self.available_frames)
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                # Flip horizontally for consistency with webcam
                frame = cv2.flip(frame, 1)
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        else:
            ret, frame = self.cap.read()
            if ret:
                # Flip horizontally 
                frame = cv2.flip(frame, 1)
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
    
    def correct_jitter_homography(self, current_frame, previous_frame):
        """Correct jitter using homography (your working method)"""
        try:
            # Convert to grayscale
            current_gray = cv2.cvtColor(cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(cv2.cvtColor(previous_frame, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
            
            # Detect features
            kp1, des1 = self.orb.detectAndCompute(previous_gray, None)
            kp2, des2 = self.orb.detectAndCompute(current_gray, None)
            
            # Skip correction if insufficient features
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                self.jitter_info = "No correction - insufficient features"
                return current_frame
            
            # Match features
            matches = self.matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Use only best matches
            good_matches = matches[:min(30, len(matches))]
            
            if len(good_matches) < 8:
                self.jitter_info = "No correction - insufficient matches"
                return current_frame
            
            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            M, mask = cv2.findHomography(dst_pts, src_pts, 
                                       cv2.RANSAC, 
                                       ransacReprojThreshold=1.5,
                                       maxIters=2000,
                                       confidence=0.99)
            
            if M is None:
                self.jitter_info = "No correction - homography failed"
                return current_frame
            
            # Check if this is small jitter (not large movement)
            tx, ty = M[0, 2], M[1, 2]
            scale_x = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
            scale_y = np.sqrt(M[1, 0]**2 + M[1, 1]**2)
            
            # Only apply for small movements
            if abs(tx) > 25 or abs(ty) > 25 or abs(scale_x - 1.0) > 0.15 or abs(scale_y - 1.0) > 0.15:
                self.jitter_info = f"Large movement - no correction (tx:{tx:.1f}, ty:{ty:.1f})"
                return current_frame
            
            # Apply correction
            h, w = current_frame.shape[:2]
            corrected_bgr = cv2.warpPerspective(
                cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR), 
                M, (w, h)
            )
            corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
            
            inliers = np.sum(mask)
            self.jitter_info = f"Corrected: tx={tx:.1f}px, ty={ty:.1f}px ({inliers}/{len(good_matches)} inliers)"
            return corrected_rgb
            
        except Exception as e:
            self.jitter_info = f"Error: {str(e)[:40]}..."
            return current_frame
    
    def detect_objects_robust_difference(self, frame1, frame2):
        """Detect objects with robust frame difference that handles residual alignment errors"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        
        # Apply slight blur to reduce noise and alignment artifacts
        gray1 = cv2.GaussianBlur(gray1, (3, 3), 0)
        gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Use higher threshold to ignore small residual differences from alignment
        threshold_value = 40  # Higher than before to ignore alignment artifacts
        _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Aggressive noise cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Additional median blur to remove small artifacts
        thresh = cv2.medianBlur(thresh, 5)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Higher minimum area to ignore small residual differences
            if area < 800:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Size filters
            if w < 20 or h < 20 or w > 350 or h > 350:
                continue
            
            # Aspect ratio filter
            aspect_ratio = w / h
            if aspect_ratio > 4 or aspect_ratio < 0.25:
                continue
            
            # Density check - ignore sparse detections
            contour_area = cv2.contourArea(contour)
            bbox_area = w * h
            density = contour_area / bbox_area
            if density < 0.3:  # Ignore very sparse detections
                continue
            
            objects.append((x, y, w, h, area, density))
        
        # Sort by area and take largest
        objects.sort(key=lambda x: x[4], reverse=True)
        return thresh, objects[:3]  # Max 3 objects
    
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
                    self.jitter_info = "Initializing..."
                    time.sleep(0.05)
                    continue
                
                # Correct jitter using your working homography method
                corrected_frame = self.correct_jitter_homography(frame_rgb, self.previous_frame)
                self.corrected_frame = corrected_frame
                
                # Detect objects with robust difference handling
                diff_image, objects = self.detect_objects_robust_difference(
                    self.previous_frame, corrected_frame
                )
                
                self.difference_image = diff_image
                self.detected_objects = objects
                
                # Update status
                num_objects = len(objects)
                self.detection_info = f"{num_objects} object{'s' if num_objects != 1 else ''} detected"
                
                # Print status
                if self.frame_count % 30 == 0 or num_objects > 0:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] Frame {self.frame_count}: {self.detection_info}")
                    print(f"  Jitter: {self.jitter_info}")
                    
                    for i, (x, y, w, h, area, density) in enumerate(objects):
                        print(f"  Object {i+1}: ({x},{y}) {w}x{h} area={area:.0f} density={density:.2f}")
                
                # Update previous frame
                self.previous_frame = frame_rgb
                
                # Control frame rate
                time.sleep(0.033)  # ~30 FPS
                
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
                ax.axis('off')
            
            # Current frame
            if self.current_frame is not None:
                self.axes[0, 0].imshow(self.current_frame)
                self.axes[0, 0].set_title('Current Frame', fontsize=10)
            
            # Jitter-corrected frame
            if self.corrected_frame is not None:
                self.axes[0, 1].imshow(self.corrected_frame)
                self.axes[0, 1].set_title('Jitter-Corrected Frame', fontsize=10)
            
            # Frame difference (should be mostly black now!)
            if self.difference_image is not None:
                self.axes[1, 0].imshow(self.difference_image, cmap='gray')
                self.axes[1, 0].set_title('Frame Difference (After Correction)', fontsize=10)
            
            # Detected objects
            if self.corrected_frame is not None:
                self.axes[1, 1].imshow(self.corrected_frame)
                
                # Draw bounding boxes
                colors = ['red', 'blue', 'green', 'yellow', 'cyan']
                for i, (x, y, w, h, area, density) in enumerate(self.detected_objects):
                    color = colors[i % len(colors)]
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                           edgecolor=color, facecolor='none')
                    self.axes[1, 1].add_patch(rect)
                    
                    # Add label with density info
                    self.axes[1, 1].text(x, y-5, f'#{i+1} ({density:.2f})', 
                                       color=color, fontweight='bold', fontsize=8,
                                       bbox=dict(boxstyle='round,pad=0.2', 
                                               facecolor='white', alpha=0.8))
                
                self.axes[1, 1].set_title('Detected Objects', fontsize=10)
            
            # Update status
            mode_info = " (Demo)" if self.demo_mode else ""
            timestamp = datetime.now().strftime("%H:%M:%S")
            status_msg = f"{timestamp} | Frame {self.frame_count} | {self.detection_info}{mode_info}"
            self.status_text.set_text(status_msg)
            
            # Turn off axes
            for ax in self.axes.flat:
                ax.axis('off')
        
        except Exception as e:
            print(f"Plot update error: {e}")
    
    def start_gui(self):
        """Start the GUI with animation"""
        try:
            # Create animation
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
    print("🎯 Refined Jitter-Corrected Tracker")
    print("Homography jitter correction + Robust difference detection")
    print("Higher thresholds to ignore alignment artifacts")
    print("Press Ctrl+C or close window to stop")
    print()
    
    try:
        tracker = RefinedTracker(demo_mode=False)
        tracker.start_gui()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    demo_mode = '--demo' in sys.argv
    
    if demo_mode:
        print("🎬 Demo Mode")
        tracker = RefinedTracker(demo_mode=True)
        tracker.start_gui()
    else:
        main() 