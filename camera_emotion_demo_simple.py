#!/usr/bin/env python3
"""
🎭 Simple Camera Emotion Demo
Face detection + emotion recognition with simulated music recommendations
"""

import cv2
import os
import sys
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCameraEmotionDemo:
    """Simple camera demo with emotion detection and music recommendations"""
    
    def __init__(self):
        # Initialize face detection
        self.face_cascade = self._setup_face_detection()
        
        # Initialize emotion detection
        try:
            from emotion_detection import get_emotion_detector
            self.emotion_detector = get_emotion_detector()
            self.emotion_available = True
            logger.info("✅ Emotion detection initialized")
        except Exception as e:
            logger.warning(f"⚠️ Emotion detection not available: {e}")
            self.emotion_available = False
        
        # Music recommendations (simulated)
        self.playlist_names = {
            "happy": "Happy Vibes 🎉",
            "sad": "Chill & Relax 🌙", 
            "neutral": "Focus Flow 🎧",
            "angry": "Energy Boost ⚡"
        }
        
        # Demo state
        self.current_emotion = None
        self.current_confidence = 0.0
        self.last_emotion_time = 0
        self.emotion_cooldown = 3.0  # seconds
        self.frame_count = 0
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'emotions_detected': {},
            'session_start': datetime.now()
        }
    
    def _setup_face_detection(self):
        """Setup face detection"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                logger.error(f"❌ Haar cascade not found: {cascade_path}")
                return None
            
            face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("✅ Face detection initialized")
            return face_cascade
            
        except Exception as e:
            logger.error(f"❌ Error setting up face detection: {e}")
            return None
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        if self.face_cascade is None:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(50, 50)
            )
            return faces
            
        except Exception as e:
            logger.error(f"❌ Error detecting faces: {e}")
            return []
    
    def detect_emotion(self, frame, face_coords):
        """Detect emotion for face region"""
        try:
            current_time = time.time()
            
            # Check cooldown
            if current_time - self.last_emotion_time < self.emotion_cooldown:
                return self.current_emotion, self.current_confidence
            
            if self.emotion_available:
                from emotion_detection import detect_emotion_from_coordinates
                emotion, confidence = detect_emotion_from_coordinates(frame, face_coords)
            else:
                # Simple fallback emotion detection
                emotion, confidence = self._simple_emotion_detection(frame, face_coords)
            
            # Update if confidence is good and emotion changed
            if confidence > 0.6 and emotion != self.current_emotion:
                self.current_emotion = emotion
                self.current_confidence = confidence
                self.last_emotion_time = current_time
                
                # Update statistics
                self.stats['total_detections'] += 1
                self.stats['emotions_detected'][emotion] = self.stats['emotions_detected'].get(emotion, 0) + 1
                
                # Simulate music recommendation
                playlist_name = self.playlist_names.get(emotion, f"{emotion.title()} Playlist")
                logger.info(f"🎭🎵 Emotion: {emotion.title()} | Recommended: {playlist_name}")
            
            return self.current_emotion, self.current_confidence
            
        except Exception as e:
            logger.error(f"❌ Error in emotion detection: {e}")
            return "neutral", 0.5
    
    def _simple_emotion_detection(self, frame, face_coords):
        """Simple fallback emotion detection"""
        try:
            x, y, w, h = face_coords
            face_region = frame[y:y+h, x:x+w]
            
            if face_region.size == 0:
                return "neutral", 0.5
            
            # Convert to grayscale
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Simple heuristics based on image properties
            mean_intensity = cv2.mean(gray_face)[0]
            
            # Basic emotion inference (very simple)
            if mean_intensity > 130:
                return "happy", 0.7
            elif mean_intensity < 90:
                return "sad", 0.6
            elif cv2.Laplacian(gray_face, cv2.CV_64F).var() > 500:
                return "angry", 0.6
            else:
                return "neutral", 0.7
                
        except Exception as e:
            logger.error(f"❌ Simple emotion detection failed: {e}")
            return "neutral", 0.5
    
    def get_emotion_emoji(self, emotion):
        """Get emoji for emotion"""
        emoji_map = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'neutral': '😐'
        }
        return emoji_map.get(emotion, '😐')
    
    def draw_overlay(self, frame, faces):
        """Draw information overlay"""
        try:
            h, w = frame.shape[:2]
            
            # Process faces
            for i, (x, y, face_w, face_h) in enumerate(faces):
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x + face_w, y + face_h), (0, 255, 0), 2)
                
                # Detect emotion
                emotion, confidence = self.detect_emotion(frame, (x, y, face_w, face_h))
                
                if emotion:
                    emoji = self.get_emotion_emoji(emotion)
                    
                    # Draw emotion info
                    emotion_text = f"{emotion.title()} {emoji}"
                    confidence_text = f"{confidence:.2f}"
                    
                    # Text background
                    text_y = y - 10
                    cv2.rectangle(frame, (x, text_y - 30), (x + face_w, text_y + 5), (0, 0, 0), -1)
                    
                    # Draw emotion text
                    cv2.putText(frame, emotion_text, (x + 5, text_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, confidence_text, (x + 5, text_y - 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Show music recommendation
                    if emotion in self.playlist_names:
                        playlist_name = self.playlist_names[emotion]
                        music_text = f"♪ {playlist_name}"
                        cv2.putText(frame, music_text, (x, y + face_h + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Main overlay panel
            overlay = frame.copy()
            panel_height = 100
            cv2.rectangle(overlay, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Title
            cv2.putText(frame, "🎭 FaceReco Emotion Demo 🎵", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Current emotion
            if self.current_emotion:
                emotion_emoji = self.get_emotion_emoji(self.current_emotion)
                current_text = f"Current: {self.current_emotion.title()} {emotion_emoji}"
                cv2.putText(frame, current_text, (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Current playlist
                if self.current_emotion in self.playlist_names:
                    playlist_text = f"♪ {self.playlist_names[self.current_emotion]}"
                    cv2.putText(frame, playlist_text, (20, 85), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Statistics
            stats_text = f"Faces: {len(faces)} | Detections: {self.stats['total_detections']}"
            cv2.putText(frame, stats_text, (w - 300, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Controls
            controls = ["ESC/Q: Quit", "S: Save", "SPACE: Reset"]
            for i, control in enumerate(controls):
                cv2.putText(frame, control, (10, h - 60 + i * 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
        except Exception as e:
            logger.error(f"❌ Error drawing overlay: {e}")
    
    def run_demo(self):
        """Run the camera demo"""
        logger.info("🎬 Starting Simple Camera Emotion Demo")
        logger.info("=" * 50)
        
        # Initialize camera
        cap = self._initialize_camera()
        if cap is None:
            return False
        
        logger.info("🎭 Demo started! Look at the camera to detect emotions")
        logger.info("📋 Controls: ESC/Q=Quit, S=Save, SPACE=Reset")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("❌ Failed to read frame")
                    break
                
                self.frame_count += 1
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Draw overlay
                self.draw_overlay(frame, faces)
                
                # Display frame
                cv2.imshow('FaceReco - Simple Emotion Demo', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("🛑 Stopping demo...")
                    break
                elif key == ord('s'):  # Save frame
                    filename = f'emotion_demo_{int(time.time())}.jpg'
                    cv2.imwrite(filename, frame)
                    logger.info(f"📸 Saved: {filename}")
                elif key == ord(' '):  # Space - reset
                    self.current_emotion = None
                    self.last_emotion_time = 0
                    logger.info("🔄 Reset emotion detection")
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Demo interrupted by user")
        
        except Exception as e:
            logger.error(f"❌ Demo error: {e}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_session_stats()
            logger.info("✅ Demo completed")
        
        return True
    
    def _initialize_camera(self):
        """Initialize camera"""
        try:
            # Try AVFoundation backend first (macOS)
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                logger.error("❌ Could not open camera")
                return None
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("✅ Camera initialized")
            return cap
            
        except Exception as e:
            logger.error(f"❌ Error initializing camera: {e}")
            return None
    
    def _print_session_stats(self):
        """Print session statistics"""
        try:
            duration = (datetime.now() - self.stats['session_start']).seconds
            
            print("\n📊 Session Statistics")
            print("=" * 30)
            print(f"Duration: {duration}s")
            print(f"Frames processed: {self.frame_count}")
            print(f"Emotion detections: {self.stats['total_detections']}")
            
            if self.stats['emotions_detected']:
                print("\n🎭 Emotions detected:")
                for emotion, count in self.stats['emotions_detected'].items():
                    emoji = self.get_emotion_emoji(emotion)
                    print(f"  {emotion.title()} {emoji}: {count}")
            
        except Exception as e:
            logger.error(f"❌ Error printing stats: {e}")

def main():
    """Main function"""
    print("🎭🎵 FaceReco Simple Camera Emotion Demo")
    print("=" * 50)
    print("Face Detection + Emotion Recognition + Music Recommendations")
    print("(Spotify integration will be added after fixing authentication)")
    print()
    
    demo = SimpleCameraEmotionDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()