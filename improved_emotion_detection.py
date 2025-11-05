#!/usr/bin/env python3
"""
🎭 Improved Emotion Detection Module
Better emotion recognition using facial feature analysis
"""

import cv2
import numpy as np
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedEmotionDetector:
    """Improved emotion detection using facial landmarks and feature analysis"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Emotion mapping
        self.emotion_emojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'neutral': '😐',
            'surprised': '😲'
        }
        
        logger.info("✅ Improved emotion detector initialized")
    
    def detect_emotion_advanced(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """
        Advanced emotion detection using multiple facial features
        """
        try:
            x, y, w, h = face_coords
            
            # Extract face region with padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return "neutral", 0.5
            
            # Convert to grayscale for analysis
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Detect facial features
            eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5, minSize=(10, 10))
            smiles = self.smile_cascade.detectMultiScale(gray_face, 1.8, 20, minSize=(25, 25))
            
            # Analyze facial features
            emotion_scores = {
                'happy': 0.0,
                'sad': 0.0,
                'angry': 0.0,
                'neutral': 0.5,  # Default baseline
                'surprised': 0.0
            }
            
            # Smile detection
            if len(smiles) > 0:
                # Strong smile detected
                smile_strength = len(smiles) * 0.3
                emotion_scores['happy'] += min(smile_strength, 0.8)
                logger.debug(f"😊 Smile detected: {len(smiles)} smiles")
            
            # Eye analysis
            if len(eyes) >= 2:
                # Both eyes detected - analyze eye region
                eye_analysis = self._analyze_eye_region(gray_face, eyes)
                emotion_scores.update(eye_analysis)
            
            # Facial geometry analysis
            geometry_analysis = self._analyze_facial_geometry(gray_face)
            for emotion, score in geometry_analysis.items():
                emotion_scores[emotion] += score * 0.3
            
            # Brightness and contrast analysis
            brightness_analysis = self._analyze_brightness_contrast(gray_face)
            for emotion, score in brightness_analysis.items():
                emotion_scores[emotion] += score * 0.2
            
            # Find dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant_emotion]
            
            # Ensure minimum confidence
            confidence = max(0.5, min(1.0, confidence))
            
            logger.debug(f"🎭 Emotion scores: {emotion_scores}")
            logger.info(f"🎭 Detected: {dominant_emotion} ({confidence:.2f})")
            
            return dominant_emotion, confidence
            
        except Exception as e:
            logger.error(f"❌ Advanced emotion detection failed: {e}")
            return "neutral", 0.5
    
    def _analyze_eye_region(self, gray_face: np.ndarray, eyes) -> dict:
        """Analyze eye region for emotion indicators"""
        scores = {'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'surprised': 0.0}
        
        try:
            if len(eyes) >= 2:
                # Sort eyes by x-coordinate (left to right)
                eyes = sorted(eyes, key=lambda e: e[0])
                
                # Analyze eye shape and position
                for ex, ey, ew, eh in eyes:
                    eye_region = gray_face[ey:ey+eh, ex:ex+ew]
                    
                    if eye_region.size > 0:
                        # Eye aspect ratio analysis
                        eye_ratio = ew / eh if eh > 0 else 1.0
                        
                        # Wide eyes might indicate surprise
                        if eye_ratio > 1.3:
                            scores['surprised'] += 0.2
                        
                        # Narrow eyes might indicate happiness (squinting from smiling)
                        elif eye_ratio < 0.8:
                            scores['happy'] += 0.1
                
        except Exception as e:
            logger.debug(f"Eye analysis error: {e}")
        
        return scores
    
    def _analyze_facial_geometry(self, gray_face: np.ndarray) -> dict:
        """Analyze facial geometry for emotion indicators"""
        scores = {'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'surprised': 0.0}
        
        try:
            h, w = gray_face.shape
            
            # Analyze upper vs lower face brightness
            upper_half = gray_face[:h//2, :]
            lower_half = gray_face[h//2:, :]
            
            upper_mean = np.mean(upper_half)
            lower_mean = np.mean(lower_half)
            
            # If lower face is brighter (smile area), might be happy
            if lower_mean > upper_mean + 10:
                scores['happy'] += 0.3
            
            # Analyze facial symmetry
            left_half = gray_face[:, :w//2]
            right_half = cv2.flip(gray_face[:, w//2:], 1)
            
            # Resize to match if needed
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            if left_half.shape == right_half.shape:
                symmetry = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0][0]
                
                # High symmetry might indicate neutral/calm state
                if symmetry > 0.8:
                    scores['neutral'] = 0.6
                # Low symmetry might indicate expression
                elif symmetry < 0.6:
                    scores['happy'] += 0.2
            
        except Exception as e:
            logger.debug(f"Geometry analysis error: {e}")
        
        return scores
    
    def _analyze_brightness_contrast(self, gray_face: np.ndarray) -> dict:
        """Analyze brightness and contrast patterns"""
        scores = {'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'surprised': 0.0}
        
        try:
            # Overall brightness
            mean_brightness = np.mean(gray_face)
            
            # Contrast (standard deviation)
            contrast = np.std(gray_face)
            
            # High contrast might indicate strong expression
            if contrast > 40:
                scores['happy'] += 0.2
                scores['angry'] += 0.1
            
            # Low contrast might indicate sad/neutral
            elif contrast < 25:
                scores['sad'] += 0.1
                scores['neutral'] += 0.1
            
            # Brightness patterns
            if mean_brightness > 130:
                scores['happy'] += 0.1
            elif mean_brightness < 90:
                scores['sad'] += 0.2
            
        except Exception as e:
            logger.debug(f"Brightness analysis error: {e}")
        
        return scores
    
    def get_emotion_emoji(self, emotion: str) -> str:
        """Get emoji for emotion"""
        return self.emotion_emojis.get(emotion, '😐')

# Global detector instance
_improved_detector = None

def get_improved_emotion_detector():
    """Get global improved emotion detector"""
    global _improved_detector
    if _improved_detector is None:
        _improved_detector = ImprovedEmotionDetector()
    return _improved_detector

def detect_emotion_improved(frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Tuple[str, float]:
    """Detect emotion using improved algorithm"""
    detector = get_improved_emotion_detector()
    return detector.detect_emotion_advanced(frame, face_coords)

if __name__ == "__main__":
    # Test the improved emotion detection
    print("🧪 Testing Improved Emotion Detection")
    detector = ImprovedEmotionDetector()
    print("✅ Improved emotion detector ready for testing")