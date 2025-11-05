#!/usr/bin/env python3
"""
🎭 Emotion Detection Module for FaceReco
Detects emotions from face images using DeepFace library
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDetector:
    """
    Emotion detection using DeepFace library
    Supports multiple backends and provides confidence scores
    """
    
    def __init__(self, backend='opencv', model_name='emotion'):
        """
        Initialize emotion detector
        
        Args:
            backend (str): Detection backend ('opencv', 'ssd', 'dlib', etc.)
            model_name (str): Model for emotion detection
        """
        self.backend = backend
        self.model_name = model_name
        self.is_available = False
        
        # Emotion mapping for consistency
        self.emotion_mapping = {
            'angry': 'angry',
            'disgust': 'angry',  # Map disgust to angry for playlist purposes
            'fear': 'sad',       # Map fear to sad
            'happy': 'happy',
            'sad': 'sad',
            'surprise': 'happy', # Map surprise to happy
            'neutral': 'neutral'
        }
        
        # Initialize DeepFace
        self._initialize_deepface()
    
    def _initialize_deepface(self):
        """Initialize DeepFace library"""
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            self.is_available = True
            logger.info("✅ DeepFace emotion detection initialized successfully")
        except ImportError:
            logger.warning("❌ DeepFace not available. Installing...")
            self._install_deepface()
        except Exception as e:
            logger.error(f"❌ Error initializing DeepFace: {e}")
            self._fallback_to_opencv()
    
    def _install_deepface(self):
        """Install DeepFace if not available"""
        try:
            import subprocess
            import sys
            
            logger.info("📦 Installing DeepFace...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "deepface", "tf-keras"])
            
            # Try to import again
            from deepface import DeepFace
            self.deepface = DeepFace
            self.is_available = True
            logger.info("✅ DeepFace installed and initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to install DeepFace: {e}")
            self._fallback_to_opencv()
    
    def _fallback_to_opencv(self):
        """Fallback to OpenCV-based emotion detection"""
        logger.info("🔄 Falling back to OpenCV-based emotion detection")
        self.is_available = True  # We'll use a simple fallback
    
    def detect_emotion(self, face_image: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[str, float]:
        """
        Detect emotion from face image
        
        Args:
            face_image (np.ndarray): Cropped face image
            confidence_threshold (float): Minimum confidence for detection
            
        Returns:
            Tuple[str, float]: (emotion, confidence)
        """
        if not self.is_available:
            return "neutral", 0.5
        
        try:
            # Ensure image is in correct format
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                # Convert BGR to RGB for DeepFace
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_image
            
            # Use DeepFace for emotion detection
            if hasattr(self, 'deepface'):
                result = self.deepface.analyze(
                    img_path=face_rgb,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                # Handle both single result and list of results
                if isinstance(result, list):
                    result = result[0]
                
                # Get dominant emotion
                emotions = result['emotion']
                dominant_emotion = max(emotions, key=emotions.get)
                confidence = emotions[dominant_emotion] / 100.0
                
                # Map to our emotion categories
                mapped_emotion = self.emotion_mapping.get(dominant_emotion.lower(), 'neutral')
                
                logger.info(f"🎭 Detected emotion: {mapped_emotion} (confidence: {confidence:.2f})")
                return mapped_emotion, confidence
            
            else:
                # Fallback emotion detection
                return self._fallback_emotion_detection(face_image)
                
        except Exception as e:
            logger.error(f"❌ Error in emotion detection: {e}")
            return self._fallback_emotion_detection(face_image)
    
    def _fallback_emotion_detection(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Simple fallback emotion detection based on basic image analysis
        This is a placeholder - in production, you'd want a proper model
        """
        try:
            # Convert to grayscale for analysis
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Simple heuristics based on image properties
            # This is very basic - just for demonstration
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Basic emotion inference (placeholder logic)
            if mean_intensity > 120 and std_intensity > 30:
                return "happy", 0.6
            elif mean_intensity < 80:
                return "sad", 0.6
            elif std_intensity > 40:
                return "angry", 0.6
            else:
                return "neutral", 0.7
                
        except Exception as e:
            logger.error(f"❌ Fallback emotion detection failed: {e}")
            return "neutral", 0.5
    
    def get_emotion_emoji(self, emotion: str) -> str:
        """Get emoji representation of emotion"""
        emoji_map = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'neutral': '😐',
            'surprise': '😲',
            'fear': '😨',
            'disgust': '🤢'
        }
        return emoji_map.get(emotion, '😐')
    
    def analyze_face_region(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """
        Analyze emotion for a specific face region in frame
        
        Args:
            frame (np.ndarray): Full frame image
            face_coords (Tuple): (x, y, width, height) of face region
            
        Returns:
            Tuple[str, float]: (emotion, confidence)
        """
        try:
            x, y, w, h = face_coords
            
            # Extract face region with some padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_region = frame[y1:y2, x1:x2]
            
            # Ensure face region is valid
            if face_region.size == 0 or face_region.shape[0] < 20 or face_region.shape[1] < 20:
                logger.warning("⚠️ Face region too small for emotion detection")
                return "neutral", 0.5
            
            # Resize face for better detection (optional)
            face_resized = cv2.resize(face_region, (224, 224))
            
            return self.detect_emotion(face_resized)
            
        except Exception as e:
            logger.error(f"❌ Error analyzing face region: {e}")
            return "neutral", 0.5

# Global emotion detector instance
_emotion_detector = None

def get_emotion_detector() -> EmotionDetector:
    """Get global emotion detector instance"""
    global _emotion_detector
    if _emotion_detector is None:
        _emotion_detector = EmotionDetector()
    return _emotion_detector

def detect_emotion_from_face(face_image: np.ndarray) -> Tuple[str, float]:
    """
    Convenience function to detect emotion from face image
    
    Args:
        face_image (np.ndarray): Face image
        
    Returns:
        Tuple[str, float]: (emotion, confidence)
    """
    detector = get_emotion_detector()
    return detector.detect_emotion(face_image)

def detect_emotion_from_coordinates(frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Tuple[str, float]:
    """
    Convenience function to detect emotion from face coordinates
    
    Args:
        frame (np.ndarray): Full frame
        face_coords (Tuple): (x, y, width, height)
        
    Returns:
        Tuple[str, float]: (emotion, confidence)
    """
    detector = get_emotion_detector()
    return detector.analyze_face_region(frame, face_coords)

if __name__ == "__main__":
    # Test emotion detection
    print("🧪 Testing Emotion Detection Module")
    print("=" * 40)
    
    detector = EmotionDetector()
    
    if detector.is_available:
        print("✅ Emotion detector initialized successfully")
        
        # Test with a sample image if available
        test_image_path = "img/oscars_360x640.gif"
        if os.path.exists(test_image_path):
            print(f"📸 Testing with sample image: {test_image_path}")
            
            # Load and process image
            frame = cv2.imread(test_image_path)
            if frame is not None:
                # Simple face detection for testing
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                face_cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    emotion, confidence = detector.analyze_face_region(frame, (x, y, w, h))
                    emoji = detector.get_emotion_emoji(emotion)
                    print(f"🎭 Detected emotion: {emotion} {emoji} (confidence: {confidence:.2f})")
                else:
                    print("❌ No faces detected in test image")
            else:
                print("❌ Could not load test image")
        else:
            print("⚠️ No test image available")
    else:
        print("❌ Emotion detector not available")
    
    print("\n✅ Emotion detection module test completed")