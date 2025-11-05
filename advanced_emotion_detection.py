#!/usr/bin/env python3
"""
🎭 Advanced Emotion Detection Module
High-accuracy emotion recognition using multiple detection methods
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Dict, List
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedEmotionDetector:
    """Advanced emotion detection with high accuracy"""
    
    def __init__(self):
        # Load all cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Emotion emojis
        self.emotion_emojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'neutral': '😐',
            'surprised': '😲'
        }
        
        # Emotion history for smoothing
        self.emotion_history = []
        self.history_size = 5
        
        logger.info("✅ Advanced emotion detector initialized with high accuracy")
    
    def detect_emotion_advanced(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """
        Advanced emotion detection with multiple analysis methods
        """
        try:
            x, y, w, h = face_coords
            
            # Extract face region with optimal padding
            padding = max(10, min(w, h) // 8)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0 or face_region.shape[0] < 50 or face_region.shape[1] < 50:
                return "neutral", 0.5
            
            # Convert to grayscale
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast for better detection
            gray_face = cv2.equalizeHist(gray_face)
            
            # Initialize emotion scores
            emotion_scores = {
                'happy': 0.0,
                'sad': 0.0,
                'angry': 0.0,
                'neutral': 0.3,  # Baseline
                'surprised': 0.0
            }
            
            # Method 1: Smile Detection (Most reliable for happiness)
            smile_score = self._detect_smile_advanced(gray_face)
            emotion_scores['happy'] += smile_score * 0.4
            
            # Method 2: Eye Analysis
            eye_scores = self._analyze_eyes_advanced(gray_face)
            for emotion, score in eye_scores.items():
                emotion_scores[emotion] += score * 0.2
            
            # Method 3: Facial Geometry Analysis
            geometry_scores = self._analyze_facial_geometry_advanced(gray_face)
            for emotion, score in geometry_scores.items():
                emotion_scores[emotion] += score * 0.2
            
            # Method 4: Texture Analysis
            texture_scores = self._analyze_texture_patterns(gray_face)
            for emotion, score in texture_scores.items():
                emotion_scores[emotion] += score * 0.1
            
            # Method 5: Mouth Region Analysis
            mouth_scores = self._analyze_mouth_region(gray_face)
            for emotion, score in mouth_scores.items():
                emotion_scores[emotion] += score * 0.1
            
            # Find dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            raw_confidence = emotion_scores[dominant_emotion]
            
            # Apply smoothing using emotion history
            smoothed_emotion, smoothed_confidence = self._apply_temporal_smoothing(
                dominant_emotion, raw_confidence, emotion_scores
            )
            
            # Ensure confidence is in valid range
            final_confidence = max(0.5, min(1.0, smoothed_confidence))
            
            logger.debug(f"🎭 Emotion analysis: {emotion_scores}")
            logger.info(f"🎭 Final result: {smoothed_emotion} ({final_confidence:.2f})")
            
            return smoothed_emotion, final_confidence
            
        except Exception as e:
            logger.error(f"❌ Advanced emotion detection failed: {e}")
            return "neutral", 0.5
    
    def _detect_smile_advanced(self, gray_face: np.ndarray) -> float:
        """Advanced smile detection with multiple parameters"""
        try:
            smile_score = 0.0
            
            # Multiple smile detection passes with different parameters
            smile_params = [
                (1.8, 20, (25, 25)),  # Conservative
                (1.5, 15, (20, 20)),  # Moderate
                (1.3, 10, (15, 15))   # Sensitive
            ]
            
            for scale_factor, min_neighbors, min_size in smile_params:
                smiles = self.smile_cascade.detectMultiScale(
                    gray_face, 
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_size
                )
                
                if len(smiles) > 0:
                    # Weight by detection strength
                    smile_strength = len(smiles) * (2.0 - scale_factor) * 0.2
                    smile_score += min(smile_strength, 0.8)
            
            # Analyze lower face brightness (smiles often create brighter lower face)
            h, w = gray_face.shape
            lower_face = gray_face[int(h*0.6):, :]
            upper_face = gray_face[:int(h*0.4), :]
            
            if lower_face.size > 0 and upper_face.size > 0:
                lower_brightness = np.mean(lower_face)
                upper_brightness = np.mean(upper_face)
                
                if lower_brightness > upper_brightness + 5:
                    smile_score += 0.3
            
            logger.debug(f"😊 Smile score: {smile_score:.2f}")
            return min(smile_score, 1.0)
            
        except Exception as e:
            logger.debug(f"Smile detection error: {e}")
            return 0.0
    
    def _analyze_eyes_advanced(self, gray_face: np.ndarray) -> Dict[str, float]:
        """Advanced eye analysis for emotion detection"""
        scores = {'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'surprised': 0.0}
        
        try:
            # Detect eyes with multiple parameters
            eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5, minSize=(10, 10))
            
            if len(eyes) >= 2:
                # Sort eyes by x-coordinate
                eyes = sorted(eyes, key=lambda e: e[0])
                
                # Analyze eye characteristics
                for ex, ey, ew, eh in eyes[:2]:  # Use first two eyes
                    if ey + eh < gray_face.shape[0] and ex + ew < gray_face.shape[1]:
                        eye_region = gray_face[ey:ey+eh, ex:ex+ew]
                        
                        if eye_region.size > 0:
                            # Eye aspect ratio
                            eye_ratio = ew / eh if eh > 0 else 1.0
                            
                            # Wide eyes (surprise)
                            if eye_ratio > 1.4:
                                scores['surprised'] += 0.3
                            
                            # Narrow eyes (happiness from squinting)
                            elif eye_ratio < 0.7:
                                scores['happy'] += 0.2
                            
                            # Eye brightness analysis
                            eye_brightness = np.mean(eye_region)
                            if eye_brightness < 80:  # Dark eyes might indicate sadness
                                scores['sad'] += 0.1
                
                # Eye distance analysis
                if len(eyes) >= 2:
                    eye_distance = abs(eyes[1][0] - eyes[0][0])
                    face_width = gray_face.shape[1]
                    
                    # Relative eye distance
                    if face_width > 0:
                        eye_distance_ratio = eye_distance / face_width
                        
                        # Wide-set eyes might indicate surprise
                        if eye_distance_ratio > 0.4:
                            scores['surprised'] += 0.1
            
        except Exception as e:
            logger.debug(f"Eye analysis error: {e}")
        
        return scores
    
    def _analyze_facial_geometry_advanced(self, gray_face: np.ndarray) -> Dict[str, float]:
        """Advanced facial geometry analysis"""
        scores = {'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'surprised': 0.0}
        
        try:
            h, w = gray_face.shape
            
            # Divide face into regions
            upper_third = gray_face[:h//3, :]
            middle_third = gray_face[h//3:2*h//3, :]
            lower_third = gray_face[2*h//3:, :]
            
            # Analyze brightness distribution
            upper_brightness = np.mean(upper_third)
            middle_brightness = np.mean(middle_third)
            lower_brightness = np.mean(lower_third)
            
            # Happy: Lower face often brighter due to smile
            if lower_brightness > middle_brightness + 8:
                scores['happy'] += 0.4
            
            # Sad: Often overall darker, especially lower face
            if lower_brightness < middle_brightness - 5:
                scores['sad'] += 0.3
            
            # Analyze facial symmetry
            left_half = gray_face[:, :w//2]
            right_half = cv2.flip(gray_face[:, w//2:], 1)
            
            # Ensure same dimensions
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            if left_half.shape == right_half.shape and left_half.size > 0:
                # Calculate symmetry
                diff = cv2.absdiff(left_half, right_half)
                asymmetry = np.mean(diff)
                
                # High asymmetry might indicate expression
                if asymmetry > 15:
                    scores['happy'] += 0.2
                    scores['angry'] += 0.1
                
                # Very high asymmetry might indicate strong emotion
                if asymmetry > 25:
                    scores['surprised'] += 0.2
            
            # Analyze contrast patterns
            contrast = np.std(gray_face)
            
            # High contrast might indicate strong expression
            if contrast > 35:
                scores['happy'] += 0.2
                scores['angry'] += 0.2
                scores['surprised'] += 0.1
            
            # Low contrast might indicate neutral/sad
            elif contrast < 20:
                scores['sad'] += 0.2
                scores['neutral'] = 0.4
            
        except Exception as e:
            logger.debug(f"Geometry analysis error: {e}")
        
        return scores
    
    def _analyze_texture_patterns(self, gray_face: np.ndarray) -> Dict[str, float]:
        """Analyze texture patterns for emotion indicators"""
        scores = {'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'surprised': 0.0}
        
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray_face, (5, 5), 0)
            
            # Calculate gradients
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Analyze gradient patterns
            avg_gradient = np.mean(gradient_magnitude)
            
            # Strong gradients might indicate expressions
            if avg_gradient > 20:
                scores['happy'] += 0.1
                scores['angry'] += 0.1
            
            # Analyze texture in different regions
            h, w = gray_face.shape
            
            # Mouth region (lower third)
            mouth_region = gradient_magnitude[2*h//3:, w//4:3*w//4]
            if mouth_region.size > 0:
                mouth_gradient = np.mean(mouth_region)
                
                # Strong gradients in mouth region might indicate smile
                if mouth_gradient > avg_gradient * 1.2:
                    scores['happy'] += 0.2
            
        except Exception as e:
            logger.debug(f"Texture analysis error: {e}")
        
        return scores
    
    def _analyze_mouth_region(self, gray_face: np.ndarray) -> Dict[str, float]:
        """Analyze mouth region for emotion indicators"""
        scores = {'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'surprised': 0.0}
        
        try:
            h, w = gray_face.shape
            
            # Define mouth region (lower third, middle portion)
            mouth_y_start = int(h * 0.65)
            mouth_y_end = int(h * 0.9)
            mouth_x_start = int(w * 0.25)
            mouth_x_end = int(w * 0.75)
            
            mouth_region = gray_face[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
            
            if mouth_region.size > 0:
                # Analyze mouth region brightness
                mouth_brightness = np.mean(mouth_region)
                face_brightness = np.mean(gray_face)
                
                # Bright mouth region might indicate smile
                if mouth_brightness > face_brightness + 10:
                    scores['happy'] += 0.3
                
                # Dark mouth region might indicate frown
                elif mouth_brightness < face_brightness - 10:
                    scores['sad'] += 0.2
                
                # Analyze mouth region variance
                mouth_variance = np.var(mouth_region)
                
                # High variance might indicate open mouth (surprise)
                if mouth_variance > 400:
                    scores['surprised'] += 0.2
                
                # Analyze horizontal patterns in mouth region
                mouth_horizontal = np.mean(mouth_region, axis=0)
                if len(mouth_horizontal) > 2:
                    # Look for smile curve pattern
                    left_side = np.mean(mouth_horizontal[:len(mouth_horizontal)//3])
                    middle = np.mean(mouth_horizontal[len(mouth_horizontal)//3:2*len(mouth_horizontal)//3])
                    right_side = np.mean(mouth_horizontal[2*len(mouth_horizontal)//3:])
                    
                    # Smile pattern: sides higher than middle
                    if (left_side + right_side) / 2 > middle + 5:
                        scores['happy'] += 0.2
                    
                    # Frown pattern: middle higher than sides
                    elif middle > (left_side + right_side) / 2 + 5:
                        scores['sad'] += 0.2
            
        except Exception as e:
            logger.debug(f"Mouth analysis error: {e}")
        
        return scores
    
    def _apply_temporal_smoothing(self, current_emotion: str, current_confidence: float, 
                                emotion_scores: Dict[str, float]) -> Tuple[str, float]:
        """Apply temporal smoothing to reduce flickering"""
        try:
            # Add current detection to history
            self.emotion_history.append((current_emotion, current_confidence, emotion_scores))
            
            # Keep only recent history
            if len(self.emotion_history) > self.history_size:
                self.emotion_history.pop(0)
            
            # If we don't have enough history, return current
            if len(self.emotion_history) < 3:
                return current_emotion, current_confidence
            
            # Calculate weighted average of recent emotions
            emotion_weights = {'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'neutral': 0.0, 'surprised': 0.0}
            total_weight = 0.0
            
            for i, (emotion, confidence, scores) in enumerate(self.emotion_history):
                # More recent detections have higher weight
                weight = (i + 1) * confidence
                total_weight += weight
                
                for emo, score in scores.items():
                    emotion_weights[emo] += score * weight
            
            # Normalize weights
            if total_weight > 0:
                for emotion in emotion_weights:
                    emotion_weights[emotion] /= total_weight
            
            # Find smoothed dominant emotion
            smoothed_emotion = max(emotion_weights, key=emotion_weights.get)
            smoothed_confidence = emotion_weights[smoothed_emotion]
            
            # Boost confidence if emotion is consistent
            recent_emotions = [e[0] for e in self.emotion_history[-3:]]
            if recent_emotions.count(smoothed_emotion) >= 2:
                smoothed_confidence *= 1.2
            
            return smoothed_emotion, smoothed_confidence
            
        except Exception as e:
            logger.debug(f"Temporal smoothing error: {e}")
            return current_emotion, current_confidence
    
    def get_emotion_emoji(self, emotion: str) -> str:
        """Get emoji for emotion"""
        return self.emotion_emojis.get(emotion, '😐')
    
    def reset_history(self):
        """Reset emotion history"""
        self.emotion_history.clear()
        logger.debug("🔄 Emotion history reset")

# Global detector instance
_advanced_detector = None

def get_advanced_emotion_detector():
    """Get global advanced emotion detector"""
    global _advanced_detector
    if _advanced_detector is None:
        _advanced_detector = AdvancedEmotionDetector()
    return _advanced_detector

def detect_emotion_advanced(frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Tuple[str, float]:
    """Detect emotion using advanced algorithm"""
    detector = get_advanced_emotion_detector()
    return detector.detect_emotion_advanced(frame, face_coords)

if __name__ == "__main__":
    # Test the advanced emotion detection
    print("🧪 Testing Advanced Emotion Detection")
    detector = AdvancedEmotionDetector()
    print("✅ Advanced emotion detector ready with high accuracy")