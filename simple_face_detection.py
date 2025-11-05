#!/usr/bin/env python3
"""
Simple face detection demo using Haar cascade on sample image
"""
import cv2
import os

print("YuNet Face Detection Demo - Haar Cascade Fallback")
print("=" * 50)

# Use Haar cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    print(f'Haar cascade not found at {cascade_path}')
    exit(1)

detector = cv2.CascadeClassifier(cascade_path)
print(f"Loaded Haar cascade from: {cascade_path}")

# Try to load the sample image
img_path = "img/oscars_360x640.gif"
if os.path.exists(img_path):
    # OpenCV can read the first frame of a GIF
    frame = cv2.imread(img_path)
    if frame is not None:
        print(f"Loaded sample image: {img_path}")
        print(f"Image dimensions: {frame.shape[1]}x{frame.shape[0]}")
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        print(f"Detected {len(faces)} faces")
        
        # Draw rectangles around faces
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {i+1}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"  Face {i+1}: ({x}, {y}) {w}x{h}")
        
        # Save the result
        output_path = "face_detection_demo_result.jpg"
        cv2.imwrite(output_path, frame)
        print(f"Result saved as: {output_path}")
        
        # Try to display the image (will work if display is available)
        try:
            cv2.imshow('Face Detection Demo', frame)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("Display not available, but result saved to file.")
            
    else:
        print("Could not load the image file")
else:
    print(f"Sample image not found: {img_path}")
    print("Creating a simple test...")
    
    # Create a simple test image
    test_img = cv2.imread(cv2.data.haarcascades + '../../../samples/data/lena.jpg')
    if test_img is not None:
        print("Using OpenCV sample image")
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        
        print(f"Detected {len(faces)} faces")
        for (x,y,w,h) in faces:
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),2)
        
        cv2.imwrite("test_face_detection.jpg", test_img)
        print("Test result saved as test_face_detection.jpg")

print("\nDemo completed!")