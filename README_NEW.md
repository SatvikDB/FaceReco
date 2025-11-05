# 🎭 YuNet Face Detection Project

A comprehensive face detection system using YuNet neural network with DepthAI hardware support and fallback options for standard cameras.

![Face Detection Demo](face_detection_demo_result.jpg)

## 🌟 Features

- **🚀 YuNet Neural Network**: State-of-the-art face detection model
- **🎥 Real-time Detection**: Live camera face detection
- **🖼️ Image Processing**: Batch processing of images
- **🔧 Multiple Backends**: DepthAI hardware, ONNX, and Haar cascade fallbacks
- **🍎 macOS Optimized**: Proper camera handling for macOS systems
- **📊 Performance Metrics**: FPS monitoring and detection statistics

## 🎯 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt
```

### Run Face Detection

#### 🎥 Real-time Camera Detection
```bash
# Primary method - Enhanced camera detection
python3 camera_face_detection.py

# Alternative - Haar cascade fallback
python3 run_haar_fallback.py --input 0
```

#### 🖼️ Image Processing
```bash
# Process sample image
python3 simple_face_detection.py

# Working demo with sample
python3 demo_working.py
```

#### 🧪 Test Camera Access
```bash
# Quick camera test
python3 quick_camera_test.py

# Comprehensive test
python3 test_camera.py
```

## 📁 Project Structure

```
├── 📋 Core Scripts
│   ├── demo.py                    # Main DepthAI demo (requires OAK hardware)
│   ├── run_haar_fallback.py       # Camera/image detection with Haar cascade
│   ├── camera_face_detection.py   # Enhanced real-time camera detection
│   └── simple_face_detection.py   # Basic image processing demo
│
├── 🧪 Testing & Utilities
│   ├── test_camera.py             # Camera access testing
│   ├── quick_camera_test.py       # Quick camera verification
│   ├── demo_working.py            # Demonstration script
│   └── test_image_detection.py    # Image detection testing
│
├── 🤖 YuNet Components
│   ├── YuNet.py                   # YuNet model implementation
│   ├── YuNetEdge.py               # Edge processing version
│   ├── YuNetRenderer.py           # Rendering utilities
│   └── FPS.py                     # Performance monitoring
│
├── 📦 Models & Assets
│   ├── models/                    # Neural network models (.blob, .onnx)
│   ├── img/                       # Sample images
│   └── requirements.txt           # Python dependencies
│
└── 📚 Documentation
    ├── README.md                  # This file
    ├── CAMERA_SETUP.md            # Camera setup guide
    └── LICENSE.txt                # License information
```

## 🎮 Usage Examples

### Real-time Face Detection
```bash
# Start camera detection with controls
python3 camera_face_detection.py

# Controls:
# - 'q' or ESC: Quit
# - 's': Save current frame
# - SPACE: Pause/Resume
```

### Image Processing
```bash
# Process specific image
python3 run_haar_fallback.py --input path/to/image.jpg

# Process sample image
python3 simple_face_detection.py
```

### DepthAI Hardware (OAK-1, OAK-D)
```bash
# Host mode processing
python3 demo.py

# Edge mode processing
python3 demo.py -e

# Synchronized processing
python3 demo.py -e -s
```

## 🔧 Configuration

### Camera Settings
- **Default Resolution**: 640x480
- **FPS**: 30
- **Detection Method**: Haar Cascade (fallback)
- **Backend**: AVFoundation (macOS), DirectShow (Windows)

### Detection Parameters
- **Scale Factor**: 1.1
- **Min Neighbors**: 5
- **Min Face Size**: 30x30 pixels
- **Score Threshold**: 0.6 (for YuNet)

## 🍎 macOS Setup

### Camera Permissions
1. Open **System Preferences** → **Security & Privacy** → **Camera**
2. Enable camera access for **Terminal** or your IDE
3. Restart terminal after granting permissions

### Troubleshooting
```bash
# Check camera permissions
system_profiler SPCameraDataType

# Reset camera permissions (requires restart)
sudo tccutil reset Camera
```

## 🚀 Performance

| Method | FPS | Accuracy | Hardware |
|--------|-----|----------|----------|
| YuNet (DepthAI) | 30+ | High | OAK-1/OAK-D |
| YuNet (ONNX) | 15-25 | High | CPU |
| Haar Cascade | 20-30 | Medium | CPU |

## 🛠️ Development

### Adding New Models
1. Place ONNX models in `models/build/`
2. Place DepthAI blobs in `models/`
3. Update model paths in configuration

### Custom Detection
```python
import cv2
from YuNet import YuNet

# Initialize detector
detector = YuNet(model_path="models/your_model.onnx")

# Process frame
faces = detector.detect(frame)
```

## 📊 Results

The system successfully detects faces with:
- ✅ **Real-time processing** at 20-30 FPS
- ✅ **Multiple face detection** in single frame
- ✅ **Landmark detection** (5-point facial landmarks)
- ✅ **Confidence scoring** for each detection

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 🙏 Acknowledgments

- **OpenCV Zoo**: Original YuNet ONNX model
- **PINTO**: Model conversion and optimization
- **DepthAI**: Hardware acceleration support
- **Luxonis**: OAK hardware platform

## 📞 Support

- 📖 Check [CAMERA_SETUP.md](CAMERA_SETUP.md) for camera issues
- 🐛 Report bugs via GitHub Issues
- 💡 Feature requests welcome!

---

**🎉 Ready to detect faces? Start with `python3 camera_face_detection.py`!**