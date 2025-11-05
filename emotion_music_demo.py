#!/usr/bin/env python3
"""
🎭🎵 FaceReco with Emotion-Based Music Recommendation
Enhanced demo that detects faces, recognizes emotions, and plays matching Spotify playlists
"""

import cv2
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import our custom modules
from emotion_detection import get_emotion_detector, detect_emotion_from_coordinates
from spotify_integration import get_spotify_controller, log_emotion_music_event

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionMusicDemo:
    """
    Main demo class that integrates face detection, emotion recognition, and music playback
    """
    
    def __init__(self, spotify_client_id: str = None, spotify_client_secret: str = None):
        """
        Initialize the emotion-music demo
        
        Args:
            spotify_client_id (str): Spotify Client ID
            spotify_client_secret (str): Spotify Client Secret
        """
        self.emotion_detector = get_emotion_detector()
        self.spotify_controller = get_spotify_controller(spotify_client_id, spotify_client_secret)
        
        # Face detection setup
        self.face_cascade = self._setup_face_detection()
        
        # Demo settings
        self.last_emotion_time = 0
        self.emotion_cooldown = 5.0  # Seconds between emotion detections
        self.current_emotion = None
        self.current_confidence = 0.0
        self.last_playlist = None
        
        # User recognition (placeholder - extend as needed)
        self.recognized_users = {}
        self.current_user = "User"
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'emotions_detected': {},
            'playlists_played': {},
            'session_start': datetime.now()
        }
    
    def _setup_face_detection(self):
        """Setup face detection cascade"""
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
    
    def process_emotion_and_music(self, frame, face_coords):
        """Process emotion detection and trigger music playback"""
        try:
            current_time = time.time()
            
            # Check cooldown to avoid too frequent emotion detection
            if current_time - self.last_emotion_time < self.emotion_cooldown:
                return self.current_emotion, self.current_confidence
            
            # Detect emotion
            emotion, confidence = detect_emotion_from_coordinates(frame, face_coords)
            
            # Update statistics
            self.stats['total_detections'] += 1
            self.stats['emotions_detected'][emotion] = self.stats['emotions_detected'].get(emotion, 0) + 1
            
            # Only proceed if confidence is high enough and emotion changed
            if confidence > 0.6 and emotion != self.current_emotion:
                self.current_emotion = emotion
                self.current_confidence = confidence
                self.last_emotion_time = current_time
                
                # Get playlist name
                playlist_name = self.spotify_controller.get_playlist_name_for_emotion(emotion)
                
                # Play music for emotion
                success = self.spotify_controller.play_playlist_for_emotion(emotion)
                
                if success:
                    self.last_playlist = playlist_name
                    self.stats['playlists_played'][emotion] = self.stats['playlists_played'].get(emotion, 0) + 1
                    
                    # Log the event
                    log_emotion_music_event(self.current_user, emotion, confidence, playlist_name)
                    
                    logger.info(f"🎭🎵 {self.current_user} | {emotion.title()} ({confidence:.2f}) | Playing: {playlist_name}")
                
            return self.current_emotion, self.current_confidence
            
        except Exception as e:
            logger.error(f"❌ Error processing emotion and music: {e}")
            return "neutral", 0.5
    
    def draw_overlay(self, frame, faces):
        """Draw information overlay on frame"""
        try:
            # Draw face rectangles and emotion info
            for i, (x, y, w, h) in enumerate(faces):
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Process emotion for this face
                emotion, confidence = self.process_emotion_and_music(frame, (x, y, w, h))
                
                # Get emotion emoji
                emoji = self.emotion_detector.get_emotion_emoji(emotion)
                
                # Draw emotion info
                emotion_text = f"{emotion.title()} {emoji}"
                confidence_text = f"{confidence:.2f}"
                
                # Draw text background
                text_y = y - 10
                cv2.rectangle(frame, (x, text_y - 25), (x + w, text_y + 5), (0, 0, 0), -1)
                
                # Draw emotion text
                cv2.putText(frame, emotion_text, (x + 5, text_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, confidence_text, (x + 5, text_y - 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw main overlay
            self._draw_main_overlay(frame)
            
        except Exception as e:
            logger.error(f"❌ Error drawing overlay: {e}")
    
    def _draw_main_overlay(self, frame):
        """Draw main information overlay"""
        try:
            h, w = frame.shape[:2]
            
            # Create semi-transparent overlay
            overlay = frame.copy()
            
            # Main info panel
            panel_height = 120
            cv2.rectangle(overlay, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Title
            cv2.putText(frame, "🎭 FaceReco + Emotion Music 🎵", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # User info
            user_text = f"User: {self.current_user}"
            cv2.putText(frame, user_text, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Current emotion
            if self.current_emotion:
                emotion_emoji = self.emotion_detector.get_emotion_emoji(self.current_emotion)
                emotion_text = f"Emotion: {self.current_emotion.title()} {emotion_emoji}"
                cv2.putText(frame, emotion_text, (20, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Current playlist
            if self.last_playlist:
                playlist_text = f"♪ Playing: {self.last_playlist}"
                cv2.putText(frame, playlist_text, (20, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Statistics panel (bottom right)
            stats_text = [
                f"Detections: {self.stats['total_detections']}",
                f"Session: {(datetime.now() - self.stats['session_start']).seconds}s"
            ]
            
            for i, text in enumerate(stats_text):
                cv2.putText(frame, text, (w - 200, h - 40 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Controls info (bottom left)
            controls = [
                "Controls:",
                "ESC/Q: Quit",
                "S: Save frame", 
                "P: Pause music",
                "R: Resume music"
            ]
            
            for i, text in enumerate(controls):
                cv2.putText(frame, text, (10, h - 100 + i * 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
        except Exception as e:
            logger.error(f"❌ Error drawing main overlay: {e}")
    
    def run_camera_demo(self, camera_id: int = 0):
        """Run the main camera demo"""
        logger.info("🎬 Starting Emotion-Music Camera Demo")
        logger.info("=" * 50)
        
        # Initialize camera
        cap = self._initialize_camera(camera_id)
        if cap is None:
            return False
        
        logger.info("🎭 Demo started! Look at the camera to detect emotions and play music")
        logger.info("📋 Controls: ESC/Q=Quit, S=Save, P=Pause, R=Resume")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("❌ Failed to read frame")
                    break
                
                frame_count += 1
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Draw overlay with emotion and music info
                self.draw_overlay(frame, faces)
                
                # Display frame
                cv2.imshow('FaceReco - Emotion Music Demo', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("🛑 Stopping demo...")
                    break
                elif key == ord('s'):  # Save frame
                    filename = f'emotion_demo_capture_{int(time.time())}.jpg'
                    cv2.imwrite(filename, frame)
                    logger.info(f"📸 Saved frame: {filename}")
                elif key == ord('p'):  # Pause music
                    self.spotify_controller.pause_playback()
                elif key == ord('r'):  # Resume music
                    self.spotify_controller.resume_playback()
                elif key == ord(' '):  # Space - reset emotion detection
                    self.current_emotion = None
                    self.last_emotion_time = 0
                    logger.info("🔄 Reset emotion detection")
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Demo interrupted by user")
        
        except Exception as e:
            logger.error(f"❌ Demo error: {e}")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self._print_session_stats()
            logger.info("✅ Demo completed")
        
        return True
    
    def _initialize_camera(self, camera_id: int):
        """Initialize camera with proper settings"""
        try:
            # Try AVFoundation backend first (macOS)
            cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                logger.error(f"❌ Could not open camera {camera_id}")
                return None
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info(f"✅ Camera {camera_id} initialized")
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
            print(f"Total detections: {self.stats['total_detections']}")
            
            if self.stats['emotions_detected']:
                print("\n🎭 Emotions detected:")
                for emotion, count in self.stats['emotions_detected'].items():
                    emoji = self.emotion_detector.get_emotion_emoji(emotion)
                    print(f"  {emotion.title()} {emoji}: {count}")
            
            if self.stats['playlists_played']:
                print("\n🎵 Playlists played:")
                for emotion, count in self.stats['playlists_played'].items():
                    playlist_name = self.spotify_controller.get_playlist_name_for_emotion(emotion)
                    print(f"  {playlist_name}: {count}")
            
        except Exception as e:
            logger.error(f"❌ Error printing stats: {e}")
    
    def run_image_demo(self, image_path: str):
        """Run demo on a static image"""
        logger.info(f"🖼️ Running emotion-music demo on image: {image_path}")
        
        try:
            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                logger.error(f"❌ Could not load image: {image_path}")
                return False
            
            # Detect faces
            faces = self.detect_faces(frame)
            logger.info(f"👥 Detected {len(faces)} faces")
            
            # Process each face
            for i, face_coords in enumerate(faces):
                emotion, confidence = detect_emotion_from_coordinates(frame, face_coords)
                emoji = self.emotion_detector.get_emotion_emoji(emotion)
                
                logger.info(f"🎭 Face {i+1}: {emotion.title()} {emoji} (confidence: {confidence:.2f})")
                
                # Play music for emotion
                playlist_name = self.spotify_controller.get_playlist_name_for_emotion(emotion)
                success = self.spotify_controller.play_playlist_for_emotion(emotion)
                
                if success:
                    logger.info(f"🎵 Playing: {playlist_name}")
            
            # Draw overlay and save result
            self.draw_overlay(frame, faces)
            
            output_path = f"emotion_music_result_{int(time.time())}.jpg"
            cv2.imwrite(output_path, frame)
            logger.info(f"💾 Result saved: {output_path}")
            
            # Display result
            cv2.imshow('Emotion-Music Demo Result', frame)
            logger.info("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in image demo: {e}")
            return False

def main():
    """Main function"""
    print("🎭🎵 FaceReco Emotion-Based Music Recommendation")
    print("=" * 50)
    
    # Check for Spotify credentials
    spotify_client_id = os.getenv('SPOTIFY_CLIENT_ID')
    spotify_client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not spotify_client_id or not spotify_client_secret:
        print("⚠️ Spotify credentials not found in environment variables")
        print("💡 Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET for full functionality")
        print("🎵 Running in demo mode (no actual music playback)")
        print()
    
    # Initialize demo
    demo = EmotionMusicDemo(spotify_client_id, spotify_client_secret)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--image" and len(sys.argv) > 2:
            # Image mode
            demo.run_image_demo(sys.argv[2])
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python3 emotion_music_demo.py                    # Camera demo")
            print("  python3 emotion_music_demo.py --image <path>     # Image demo")
            print("  python3 emotion_music_demo.py --help             # Show help")
        else:
            print("❌ Invalid arguments. Use --help for usage info.")
    else:
        # Camera mode (default)
        demo.run_camera_demo()

if __name__ == "__main__":
    main()