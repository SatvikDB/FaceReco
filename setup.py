#!/usr/bin/env python3
"""
🎭🎵 FaceReco Enhanced Setup Script
Setup for face detection + emotion recognition + Spotify music integration
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ required")
        return False
    
    print("✅ Python version compatible")
    return True

def install_basic_dependencies():
    """Install basic required Python packages"""
    print("📦 Installing basic dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install basic requirements
    basic_packages = [
        "opencv-python>=4.5.1.48",
        "numpy>=1.21.0",
        "Pillow>=9.0.0"
    ]
    
    for package in basic_packages:
        if not run_command(f"{sys.executable} -m pip install '{package}'", f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")
    
    return True

def install_emotion_dependencies():
    """Install emotion detection dependencies"""
    print("\n🎭 Installing emotion detection dependencies...")
    
    emotion_packages = [
        "deepface>=0.0.75",
        "tensorflow>=2.12.0",
        "tf-keras>=2.12.0"
    ]
    
    success = True
    for package in emotion_packages:
        if not run_command(f"{sys.executable} -m pip install '{package}'", f"Installing {package}"):
            print(f"⚠️ Failed to install {package}")
            success = False
    
    return success

def install_spotify_dependencies():
    """Install Spotify integration dependencies"""
    print("\n🎵 Installing Spotify integration dependencies...")
    
    return run_command(f"{sys.executable} -m pip install 'spotipy>=2.22.1'", "Installing spotipy")

def test_basic_installation():
    """Test basic installation"""
    print("\n🧪 Testing basic installation...")
    
    try:
        import cv2
        import numpy as np
        print(f"✅ OpenCV version: {cv2.__version__}")
        print(f"✅ NumPy version: {np.__version__}")
        
        # Test Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            print("✅ Haar cascade available")
        else:
            print("❌ Haar cascade not found")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Basic import error: {e}")
        return False

def test_emotion_installation():
    """Test emotion detection installation"""
    print("\n🧪 Testing emotion detection...")
    
    try:
        from emotion_detection import get_emotion_detector
        detector = get_emotion_detector()
        
        if detector.is_available:
            print("✅ Emotion detection available")
            return True
        else:
            print("⚠️ Emotion detection in fallback mode")
            return True
            
    except ImportError as e:
        print(f"⚠️ Emotion detection not available: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Emotion detection error: {e}")
        return False

def test_spotify_installation():
    """Test Spotify integration installation"""
    print("\n🧪 Testing Spotify integration...")
    
    try:
        from spotify_integration import get_spotify_controller
        controller = get_spotify_controller()
        print("✅ Spotify integration available (demo mode)")
        return True
        
    except ImportError as e:
        print(f"⚠️ Spotify integration not available: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Spotify integration error: {e}")
        return False

def setup_configuration():
    """Setup configuration files"""
    print("\n⚙️ Setting up configuration...")
    
    # Check if Spotify config exists
    if not os.path.exists("spotify_config.json"):
        print("💡 Spotify not configured yet")
        setup_spotify = input("Set up Spotify integration now? (y/n): ").lower().strip()
        
        if setup_spotify == 'y':
            try:
                from spotify_config import main as spotify_setup
                spotify_setup()
            except Exception as e:
                print(f"⚠️ Spotify setup failed: {e}")
                print("💡 You can run 'python3 spotify_config.py' later")
    
    return True

def main():
    """Main setup function"""
    print("🎭🎵 FaceReco Enhanced Setup")
    print("Face Detection + Emotion Recognition + Spotify Music")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install basic dependencies
    if not install_basic_dependencies():
        print("❌ Failed to install basic dependencies")
        sys.exit(1)
    
    # Test basic installation
    if not test_basic_installation():
        print("❌ Basic installation test failed")
        sys.exit(1)
    
    print("\n✅ Basic face detection setup completed!")
    
    # Ask about enhanced features
    print("\n🚀 Enhanced Features Setup")
    print("=" * 30)
    
    install_enhanced = input("Install emotion detection and Spotify integration? (y/n): ").lower().strip()
    
    if install_enhanced == 'y':
        # Install emotion dependencies
        emotion_success = install_emotion_dependencies()
        
        # Install Spotify dependencies  
        spotify_success = install_spotify_dependencies()
        
        # Test enhanced features
        if emotion_success:
            test_emotion_installation()
        
        if spotify_success:
            test_spotify_installation()
        
        # Setup configuration
        if emotion_success and spotify_success:
            setup_configuration()
    
    print("\n🎉 Setup completed!")
    print("=" * 20)
    
    print("\n🚀 Available demos:")
    print("  python3 demo_working.py                # Enhanced demo with emotion")
    print("  python3 simple_face_detection.py      # Basic face detection")
    print("  python3 camera_face_detection.py      # Real-time camera detection")
    
    if install_enhanced == 'y':
        print("  python3 emotion_music_demo.py          # Full emotion + music demo")
        print("  python3 spotify_config.py              # Configure Spotify")
    
    print("\n📚 Documentation:")
    print("  README.md                              # Full project documentation")
    print("  CAMERA_SETUP.md                       # Camera setup guide")
    
    print(f"\n💡 Next steps:")
    if install_enhanced == 'y':
        print("1. Configure Spotify: python3 spotify_config.py")
        print("2. Run full demo: python3 emotion_music_demo.py")
    else:
        print("1. Test camera: python3 camera_face_detection.py")
        print("2. For enhanced features, re-run setup and choose 'y'")

if __name__ == "__main__":
    main()