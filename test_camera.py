#!/usr/bin/env python3
"""
Simple camera test to check if camera is accessible
"""
import cv2
import sys

print("Testing camera access...")

# Test different camera backends
backends = [
    ("Default", None),
    ("AVFoundation (macOS)", cv2.CAP_AVFOUNDATION),
]

for name, backend in backends:
    print(f"\nTrying {name}...")
    
    try:
        if backend is None:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(0, backend)
        
        if cap.isOpened():
            print(f"✅ Camera opened with {name}")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"✅ Successfully read frame: {frame.shape}")
                
                # Save a test image
                cv2.imwrite("camera_test.jpg", frame)
                print("✅ Test image saved as camera_test.jpg")
                
                cap.release()
                print("🎉 Camera is working!")
                break
            else:
                print("❌ Could not read frame")
        else:
            print(f"❌ Could not open camera with {name}")
        
        cap.release()
        
    except Exception as e:
        print(f"❌ Error with {name}: {e}")

else:
    print("\n❌ Camera not accessible")
    print("💡 Possible solutions:")
    print("   1. Check System Preferences > Security & Privacy > Camera")
    print("   2. Grant camera access to Terminal or Python")
    print("   3. Close other apps that might be using the camera")
    print("   4. Try running with sudo (not recommended but might work)")

print("\nDone.")