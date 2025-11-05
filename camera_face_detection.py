#!/usr/bin/env python3
"""
Real-time face detection using laptop camera
Optimized for macOS
"""
import cv2
import sys

def main():
    print("🎥 Starting Camera Face Detection")
    print("=" * 40)
    
    # Load Haar cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not cv2.os.path.exists(cascade_path):
        print(f'❌ Haar cascade not found at {cascade_path}')
        return False
    
    detector = cv2.CascadeClassifier(cascade_path)
    print(f"✅ Loaded Haar cascade detector")
    
    # Try to open camera (try different backends for macOS)
    print("🔍 Attempting to open camera...")
    
    # Try different camera indices and backends
    camera_opened = False
    cap = None
    
    for camera_id in [0, 1, 2]:
        print(f"  Trying camera {camera_id}...")
        
        # Try AVFoundation backend (best for macOS)
        try:
            cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                print(f"✅ Camera {camera_id} opened with AVFoundation")
                camera_opened = True
                break
        except:
            pass
        
        # Try default backend
        try:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                print(f"✅ Camera {camera_id} opened with default backend")
                camera_opened = True
                break
        except:
            pass
        
        if cap:
            cap.release()
    
    if not camera_opened:
        print("❌ Could not open any camera")
        print("💡 Make sure:")
        print("   - Camera is not being used by another app")
        print("   - Camera permissions are granted to Terminal/Python")
        print("   - Try running: sudo python3 camera_face_detection.py")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("🎬 Camera ready! Starting face detection...")
    print("📋 Controls:")
    print("   - Press 'q' or ESC to quit")
    print("   - Press 's' to save current frame")
    print("   - Press SPACE to pause/resume")
    
    frame_count = 0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                frame_count += 1
                
                # Convert to grayscale for detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = detector.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Draw rectangles around faces
                for i, (x, y, w, h) in enumerate(faces):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f'Face {i+1}', (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add info overlay
                info_text = f"Faces: {len(faces)} | Frame: {frame_count}"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if paused:
                    cv2.putText(frame, "PAUSED - Press SPACE to resume", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Face Detection - Camera', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("🛑 Stopping...")
                break
            elif key == ord('s'):  # Save frame
                filename = f'camera_capture_{frame_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"📸 Saved frame as {filename}")
            elif key == ord(' '):  # Space - pause/resume
                paused = not paused
                print(f"⏸️  {'Paused' if paused else 'Resumed'}")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released and windows closed")
        print(f"📊 Total frames processed: {frame_count}")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 Face detection completed successfully!")
    else:
        print("❌ Face detection failed to start")
        sys.exit(1)