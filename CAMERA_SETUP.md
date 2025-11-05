# 🎥 Camera Setup Guide for Face Detection

## Current Status
✅ **Face Detection Working**: Successfully tested with sample images  
❌ **Camera Access**: Blocked by macOS privacy settings

## 🔧 Enable Camera Access on macOS

### Method 1: System Preferences
1. Open **System Preferences** (or **System Settings** on newer macOS)
2. Go to **Security & Privacy** → **Privacy** → **Camera**
3. Look for **Terminal** or **Python** in the list
4. Check the box to enable camera access
5. If not listed, click the **+** button and add:
   - `/Applications/Utilities/Terminal.app`
   - `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`

### Method 2: Command Line Check
```bash
# Check camera permissions
system_profiler SPCameraDataType

# Reset camera permissions (requires restart)
sudo tccutil reset Camera
```

### Method 3: Alternative Terminal
Try running from a different terminal:
- **iTerm2** (if installed)
- **VS Code Terminal**
- **PyCharm Terminal**

## 🚀 Test Camera Access

### Quick Test
```bash
python3 test_camera.py
```

### Full Face Detection
```bash
# With camera
python3 run_haar_fallback.py --input 0

# With sample image (always works)
python3 run_haar_fallback.py --input img/oscars_360x640.gif
```

## 🎯 Available Demo Options

### 1. ✅ Image-based Detection (Working)
```bash
python3 simple_face_detection.py
```
- Uses sample image from `img/oscars_360x640.gif`
- Detects faces and saves result
- No camera permissions needed

### 2. 🎥 Real-time Camera Detection
```bash
python3 camera_face_detection.py
```
- Requires camera permissions
- Real-time face detection
- Interactive controls (save, pause, quit)

### 3. 🔄 Fallback Script
```bash
python3 run_haar_fallback.py --input 0        # Camera
python3 run_haar_fallback.py --input image.jpg # Image
```

## 🛠️ Troubleshooting

### Camera Not Working?
1. **Close other apps** using camera (Zoom, Skype, etc.)
2. **Restart Terminal** after granting permissions
3. **Try different camera IDs**: 0, 1, 2
4. **Check if camera LED turns on** when running script

### Still Not Working?
Use the image-based demo which is fully functional:
```bash
python3 simple_face_detection.py
```

## 📊 Project Status Summary

| Feature | Status | Notes |
|---------|--------|-------|
| Face Detection Algorithm | ✅ Working | Haar cascade implementation |
| Image Processing | ✅ Working | Successfully processes sample images |
| Result Visualization | ✅ Working | Saves annotated images |
| Camera Access | ⚠️ Needs Setup | macOS privacy settings |
| YuNet ONNX Model | ❌ Compatibility Issue | OpenCV version conflict |
| DepthAI Hardware | ❌ Not Available | Requires OAK-1/OAK-D device |

The core face detection functionality is working perfectly! 🎉