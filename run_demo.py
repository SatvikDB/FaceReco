#!/usr/bin/env python3
"""
YuNet Face Detection Project Demo
This demo shows face detection capabilities using available methods.
"""
import cv2
import os
import sys

def main():
    print("🎭 YuNet Face Detection Project Demo")
    print("=" * 50)
    print()
    
    # Check OpenCV version
    print(f"OpenCV Version: {cv2.__version__}")
    print()
    
    # Method 1: Try YuNet with ONNX (if supported)
    print("🔍 Method 1: YuNet ONNX Model")
    print("-" * 30)
    
    model_path = "models/build/face_detection_yunet.onnx"
    if os.path.exists(model_path):
        print(f"✓ Found YuNet ONNX model: {model_path}")
        
        if hasattr(cv2, 'FaceDetectorYN_create'):
            try:
                detector = cv2.FaceDetectorYN_create(model_path, "", (180, 320))
                print("✓ YuNet detector created successfully!")
                yunet_available = True
            except Exception as e:
                print(f"✗ YuNet creation failed: {str(e)[:100]}...")
                yunet_available = False
        else:
            print("✗ FaceDetectorYN not available in this OpenCV build")
            yunet_available = False
    else:
        print(f"✗ YuNet ONNX model not found: {model_path}")
        yunet_available = False
    
    print()
    
    # Method 2: Haar Cascade (fallback)
    print("🔍 Method 2: Haar Cascade (Fallback)")
    print("-" * 30)
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if os.path.exists(cascade_path):
        haar_detector = cv2.CascadeClassifier(cascade_path)
        print(f"✓ Haar cascade loaded: {cascade_path}")
        haar_available = True
    else:
        print(f"✗ Haar cascade not found: {cascade_path}")
        haar_available = False
    
    print()
    
    # Test with sample image
    print("🖼️  Testing with Sample Image")
    print("-" * 30)
    
    img_path = "img/oscars_360x640.gif"
    if os.path.exists(img_path):
        frame = cv2.imread(img_path)
        if frame is not None:
            print(f"✓ Loaded sample image: {img_path}")
            print(f"  Image dimensions: {frame.shape[1]}x{frame.shape[0]}")
            
            results = []
            
            # Test YuNet if available
            if yunet_available:
                try:
                    yunet_frame = frame.copy()
                    faces = detector.detect(yunet_frame)
                    if isinstance(faces, tuple) and len(faces) == 2:
                        faces = faces[1]
                    
                    yunet_count = 0
                    if faces is not None and len(faces) > 0:
                        for f in faces:
                            x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                            score = f[14]
                            if score >= 0.6:
                                yunet_count += 1
                                cv2.rectangle(yunet_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                # Draw landmarks
                                for li in range(5):
                                    lx = int(f[4 + li * 2])
                                    ly = int(f[4 + li * 2 + 1])
                                    cv2.circle(yunet_frame, (lx, ly), 2, (0, 0, 255), -1)
                                cv2.putText(yunet_frame, f"YuNet {score:.2f}", (x, y - 6), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    cv2.imwrite("yunet_result.jpg", yunet_frame)
                    results.append(f"YuNet: {yunet_count} faces detected → yunet_result.jpg")
                    
                except Exception as e:
                    results.append(f"YuNet: Failed - {str(e)[:50]}...")
            
            # Test Haar cascade
            if haar_available:
                haar_frame = frame.copy()
                gray = cv2.cvtColor(haar_frame, cv2.COLOR_BGR2GRAY)
                faces = haar_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                for i, (x, y, w, h) in enumerate(faces):
                    cv2.rectangle(haar_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(haar_frame, f"Haar {i+1}", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.imwrite("haar_result.jpg", haar_frame)
                results.append(f"Haar Cascade: {len(faces)} faces detected → haar_result.jpg")
            
            # Show results
            print()
            for result in results:
                print(f"  ✓ {result}")
            
        else:
            print(f"✗ Could not load image: {img_path}")
    else:
        print(f"✗ Sample image not found: {img_path}")
    
    print()
    
    # Show available models
    print("📁 Available Models")
    print("-" * 30)
    
    blob_models = []
    if os.path.exists("models"):
        for file in os.listdir("models"):
            if file.endswith(".blob"):
                blob_models.append(file)
    
    if blob_models:
        print("  DepthAI/OAK Models (.blob):")
        for model in sorted(blob_models)[:5]:  # Show first 5
            print(f"    • {model}")
        if len(blob_models) > 5:
            print(f"    ... and {len(blob_models) - 5} more")
    
    if os.path.exists("models/build/face_detection_yunet.onnx"):
        print("  ONNX Models:")
        print("    • face_detection_yunet.onnx")
    
    print()
    
    # Summary
    print("📋 Summary")
    print("-" * 30)
    print("This YuNet face detection project supports:")
    print("  • DepthAI/OAK hardware with optimized blob models")
    print("  • CPU inference with ONNX models (when compatible)")
    print("  • Haar cascade fallback for basic face detection")
    print()
    print("For full functionality, use with DepthAI hardware (OAK-1, OAK-D, etc.)")
    print("Current demo shows fallback capabilities on your system.")
    
    print()
    print("🎉 Demo completed! Check the generated result images.")

if __name__ == "__main__":
    main()