#!/usr/bin/env python3
"""
Test YuNet face detection on a static image
"""
import cv2
import os
import sys

# Check if we have the ONNX model
model_path = "models/build/face_detection_yunet.onnx"
if not os.path.isfile(model_path):
    print(f"Model not found: {model_path}")
    sys.exit(1)

print("Using model:", model_path)

# Check FaceDetectorYN availability
if not hasattr(cv2, 'FaceDetectorYN_create'):
    print("Your OpenCV build doesn't expose FaceDetectorYN_create().")
    print("Falling back to Haar cascade...")
    
    # Fallback to Haar cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(cascade_path)
    
    # Load a test image (create a simple one if needed)
    print("Creating a test image with face detection using Haar cascade...")
    
    # Try to load the gif as an image (OpenCV can read first frame)
    img_path = "img/oscars_360x640.gif"
    if os.path.exists(img_path):
        frame = cv2.imread(img_path)
        if frame is not None:
            print(f"Loaded image: {img_path}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
            
            print(f"Detected {len(faces)} faces")
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            
            # Save result
            cv2.imwrite("face_detection_result.jpg", frame)
            print("Result saved as face_detection_result.jpg")
        else:
            print("Could not load the image")
    else:
        print("Sample image not found")
    
    sys.exit(0)

# Create YuNet detector
net_input_size = (320, 180)
try:
    detector = cv2.FaceDetectorYN_create(model_path, "", (net_input_size[1], net_input_size[0]))
    print("YuNet detector created successfully!")
except Exception as e:
    print("Failed to create FaceDetectorYN:", e)
    sys.exit(1)

# Try to load and process the sample image
img_path = "img/oscars_360x640.gif"
if os.path.exists(img_path):
    frame = cv2.imread(img_path)
    if frame is not None:
        print(f"Loaded image: {img_path}")
        
        # Detect faces
        faces = detector.detect(frame)
        if isinstance(faces, tuple) and len(faces) == 2:
            faces = faces[1]
        
        if faces is not None and len(faces) > 0:
            print(f"Detected {len(faces)} faces with YuNet")
            for f in faces:
                x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                score = f[14]
                if score < 0.6:
                    continue
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Draw landmarks (5 points)
                for li in range(5):
                    lx = int(f[4 + li * 2])
                    ly = int(f[4 + li * 2 + 1])
                    cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)
                cv2.putText(frame, f"{score:.2f}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        else:
            print("No faces detected")
        
        # Save result
        cv2.imwrite("yunet_detection_result.jpg", frame)
        print("Result saved as yunet_detection_result.jpg")
    else:
        print("Could not load the image")
else:
    print("Sample image not found")