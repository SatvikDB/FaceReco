#!/usr/bin/env python3
import cv2
import sys

print("Testing camera access...")
sys.stdout.flush()

try:
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        print("✅ Camera opened successfully!")
        sys.stdout.flush()
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret:
            print(f"✅ Frame captured: {frame.shape}")
            sys.stdout.flush()
            
            # Quick face detection test
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            detector = cv2.CascadeClassifier(cascade_path)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
            
            print(f"🎯 Detected {len(faces)} faces in first frame!")
            sys.stdout.flush()
            
            # Draw rectangles and save
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            
            cv2.imwrite("camera_test_result.jpg", frame)
            print("📸 Test result saved as camera_test_result.jpg")
            sys.stdout.flush()
            
        else:
            print("❌ Could not read frame from camera")
            sys.stdout.flush()
        
        cap.release()
        
    else:
        print("❌ Could not open camera")
        print("💡 Check camera permissions in System Preferences > Security & Privacy > Camera")
        sys.stdout.flush()

except Exception as e:
    print(f"❌ Error: {e}")
    sys.stdout.flush()

print("Test completed.")
sys.stdout.flush()