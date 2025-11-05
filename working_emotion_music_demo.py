#!/usr/bin/env python3
"""
🎭🎵 Working Emotion-Music Demo
Improved emotion detection + Working Spotify integration
"""

import cv2
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import our modules
from improved_emotion_detection import get_improved_emotion_detector, detect_emotion_improved
from simple_spotify_auth import setup_spotify_simple, play_spotify_playlist

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkingEmotionMusicDemo:
    """Working demo with improved emotion detection and Spotify integration"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        
        # Initialize components
        self.face_cascade = self._setup_face_detection()
        self.emotion_detector = get_improved_emotion_detector()
        self.spotify_client = None
        
        # Demo state
        self.current_emotion = None
        self.current_confidence = 0.0
        self.last_emotion_time = 0
        self.last_music_time = 0
        self.emotion_cooldown = 2.0
        self.music_cooldown = 10.0  # Prevent too frequent playlist changes
        self.frame_count = 0
        self.is_spotify_connected = False
        
        # Playlist mapping
        self.playlist_names = {
            "happy": "Happy Vibes 🎉",
            "sad": "Chill & Relax 🌙", 
            "neutral": "Focus Flow 🎧",
            "angry": "Energy Boost ⚡",
            "surprised": "Surprise Mix 🎊"
        }
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'emotions_detected': {},
            'playlists_played': {},
            'session_start': datetime.now()
        }
        
        logger.info("✅ Working emotion-music demo initialized")
    
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
    
    def setup_spotify(self):
        """Setup Spotify connection"""
        try:
            logger.info("🎵 Setting up Spotify connection...")
            
            print("\n🎵 Spotify Setup")
            print("=" * 20)
            print("Setting up Spotify authentication...")
            print("A browser window will open for login.")
            print()
            
            self.spotify_client = setup_spotify_simple(self.client_id, self.client_secret)
            
            if self.spotify_client:
                self.is_spotify_connected = True
                logger.info("✅ Spotify connected successfully!")
                
                # Test with a quick playlist check
                try:
                    devices = self.spotify_client.devices()
                    if devices['devices']:
                        logger.info(f"🎧 Available device: {devices['devices'][0]['name']}")
                    else:
                        logger.warning("⚠️ No active Spotify devices. Please open Spotify on a device.")
                except:
                    pass
                
                return True
            else:
                logger.warning("⚠️ Spotify connection failed, running in demo mode")
                return False
                
        except Exception as e:
            logger.error(f"❌ Spotify setup error: {e}")
            return False
    
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
                minSize=(80, 80)
            )
            return faces
            
        except Exception as e:
            logger.error(f"❌ Error detecting faces: {e}")
            return []
    
    def process_emotion_and_music(self, frame, face_coords):
        """Process emotion detection and trigger music playback"""
        try:
            current_time = time.time()
            
            # Check emotion detection cooldown
            if current_time - self.last_emotion_time < self.emotion_cooldown:
                return self.current_emotion, self.current_confidence
            
            # Detect emotion using improved algorithm
            emotion, confidence = detect_emotion_improved(frame, face_coords)
            
            # Update statistics
            self.stats['total_detections'] += 1
            self.stats['emotions_detected'][emotion] = self.stats['emotions_detected'].get(emotion, 0) + 1
            
            # Only proceed if confidence is high enough
            if confidence > 0.6:
                # Update current emotion
                emotion_changed = emotion != self.current_emotion
                self.current_emotion = emotion
                self.current_confidence = confidence
                self.last_emotion_time = current_time
                
                # Play music if emotion changed and enough time passed
                if (emotion_changed and 
                    current_time - self.last_music_time > self.music_cooldown):
                    
                    if self.is_spotify_connected:
                        success = play_spotify_playlist(self.spotify_client, emotion)
                        if success:
                            self.last_music_time = current_time
                            self.stats['playlists_played'][emotion] = self.stats['playlists_played'].get(emotion, 0) + 1
                            playlist_name = self.playlist_names.get(emotion, f"{emotion.title()} Playlist")
                            logger.info(f"🎭🎵 {emotion.title()} ({confidence:.2f}) | ♪ Playing: {playlist_name}")
                        else:
                            logger.warning(f"⚠️ Failed to play {emotion} playlist")
                    else:
                        # Demo mode
                        playlist_name = self.playlist_names.get(emotion, f"{emotion.title()} Playlist")
                        logger.info(f"🎭🎵 [DEMO] {emotion.title()} ({confidence:.2f}) | Would play: {playlist_name}")
            
            return self.current_emotion, self.current_confidence
            
        except Exception as e:
            logger.error(f"❌ Error processing emotion and music: {e}")
            return "neutral", 0.5
    
    def draw_enhanced_overlay(self, frame, faces):
        """Draw enhanced information overlay"""
        try:
            h, w = frame.shape[:2]
            
            # Process faces
            for i, (x, y, face_w, face_h) in enumerate(faces):
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x + face_w, y + face_h), (0, 255, 0), 3)
                
                # Process emotion
                emotion, confidence = self.process_emotion_and_music(frame, (x, y, face_w, face_h))
                
                if emotion:
                    emoji = self.emotion_detector.get_emotion_emoji(emotion)
                    
                    # Draw emotion info
                    emotion_text = f"{emotion.title()} {emoji}"
                    confidence_text = f"{confidence:.2f}"
                    
                    # Text background
                    text_y = y - 15
                    cv2.rectangle(frame, (x, text_y - 45), (x + face_w, text_y + 5), (0, 0, 0), -1)
                    
                    # Draw emotion text
                    cv2.putText(frame, emotion_text, (x + 5, text_y - 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, confidence_text, (x + 5, text_y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Show music info
                    if emotion in self.playlist_names:
                        playlist_name = self.playlist_names[emotion]
                        music_text = f"♪ {playlist_name}"
                        cv2.putText(frame, music_text, (x, y + face_h + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Main info panel
            self._draw_main_info_panel(frame, faces)
            
        except Exception as e:
            logger.error(f"❌ Error drawing overlay: {e}")
    
    def _draw_main_info_panel(self, frame, faces):
        """Draw main information panel"""
        try:
            h, w = frame.shape[:2]
            
            # Create overlay
            overlay = frame.copy()
            panel_height = 140
            cv2.rectangle(overlay, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Title
            cv2.putText(frame, "🎭🎵 FaceReco Working Demo", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Spotify status
            spotify_status = "🎵 Spotify: Connected ✅" if self.is_spotify_connected else "🎵 Spotify: Demo Mode ⚠️"
            color = (0, 255, 0) if self.is_spotify_connected else (0, 255, 255)
            cv2.putText(frame, spotify_status, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Current emotion
            if self.current_emotion:
                emotion_emoji = self.emotion_detector.get_emotion_emoji(self.current_emotion)
                current_text = f"Emotion: {self.current_emotion.title()} {emotion_emoji}"
                cv2.putText(frame, current_text, (20, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Current playlist
                if self.current_emotion in self.playlist_names:
                    playlist_text = f"♪ {self.playlist_names[self.current_emotion]}"
                    cv2.putText(frame, playlist_text, (20, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Face count and stats
            stats_text = f"Faces: {len(faces)} | Detections: {self.stats['total_detections']}"
            cv2.putText(frame, stats_text, (20, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Controls
            controls = [
                "Controls: ESC/Q=Quit, S=Save, SPACE=Reset",
                "P=Pause Music, R=Resume Music"
            ]
            
            for i, text in enumerate(controls):
                cv2.putText(frame, text, (10, h - 30 + i * 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
        except Exception as e:
            logger.error(f"❌ Error drawing main panel: {e}")
    
    def run_working_demo(self):
        """Run the working emotion-music demo"""
        logger.info("🎬 Starting Working Emotion-Music Demo")
        logger.info("=" * 50)
        
        # Setup Spotify
        spotify_setup = input("Setup Spotify integration? (y/n): ").lower().strip()
        if spotify_setup == 'y':
            self.setup_spotify()
        else:
            logger.info("🎵 Running in demo mode (no Spotify)")
        
        # Initialize camera
        cap = self._initialize_camera()
        if cap is None:
            return False
        
        logger.info("🎭 Working demo started!")
        logger.info("📋 Try different expressions - smile, frown, angry face!")
        if self.is_spotify_connected:
            logger.info("🎵 Music will play automatically based on your emotions!")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("❌ Failed to read frame")
                    break
                
                self.frame_count += 1
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Draw enhanced overlay
                self.draw_enhanced_overlay(frame, faces)
                
                # Display frame
                cv2.imshow('FaceReco - Working Emotion Music Demo', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("🛑 Stopping demo...")
                    break
                elif key == ord('s'):  # Save frame
                    filename = f'working_demo_{int(time.time())}.jpg'
                    cv2.imwrite(filename, frame)
                    logger.info(f"📸 Saved: {filename}")
                elif key == ord('p'):  # Pause music
                    if self.is_spotify_connected and self.spotify_client:
                        try:
                            self.spotify_client.pause_playback()
                            logger.info("⏸️ Music paused")
                        except Exception as e:
                            logger.warning(f"⚠️ Pause failed: {e}")
                elif key == ord('r'):  # Resume music
                    if self.is_spotify_connected and self.spotify_client:
                        try:
                            self.spotify_client.start_playback()
                            logger.info("▶️ Music resumed")
                        except Exception as e:
                            logger.warning(f"⚠️ Resume failed: {e}")
                elif key == ord(' '):  # Space - reset
                    self.current_emotion = None
                    self.last_emotion_time = 0
                    self.last_music_time = 0
                    logger.info("🔄 Reset emotion detection")
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Demo interrupted by user")
        
        except Exception as e:
            logger.error(f"❌ Demo error: {e}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_session_stats()
            logger.info("✅ Working demo finished")
        
        return True
    
    def _initialize_camera(self):
        """Initialize camera"""
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                logger.error("❌ Could not open camera")
                return None
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
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
            
            print("\n📊 Working Demo Session Statistics")
            print("=" * 40)
            print(f"Duration: {duration}s")
            print(f"Frames processed: {self.frame_count}")
            print(f"Emotion detections: {self.stats['total_detections']}")
            print(f"Spotify status: {'Connected ✅' if self.is_spotify_connected else 'Demo Mode ⚠️'}")
            
            if self.stats['emotions_detected']:
                print("\n🎭 Emotions detected:")
                for emotion, count in self.stats['emotions_detected'].items():
                    emoji = self.emotion_detector.get_emotion_emoji(emotion)
                    print(f"  {emotion.title()} {emoji}: {count}")
            
            if self.stats['playlists_played']:
                print("\n🎵 Playlists played:")
                for emotion, count in self.stats['playlists_played'].items():
                    playlist_name = self.playlist_names[emotion]
                    print(f"  {playlist_name}: {count}")
            
        except Exception as e:
            logger.error(f"❌ Error printing stats: {e}")

def main():
    """Main function"""
    print("🎭🎵 FaceReco Working Emotion-Music Demo")
    print("=" * 50)
    print("Improved Emotion Detection + Working Spotify Integration")
    print()
    
    # Get Spotify credentials
    try:
        from spotify_credentials import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
        client_id = SPOTIFY_CLIENT_ID
        client_secret = SPOTIFY_CLIENT_SECRET
    except ImportError:
        print("❌ Spotify credentials not found")
        print("Please make sure spotify_credentials.py exists with your credentials")
        return
    
    if not client_id or not client_secret or "YOUR_CLIENT" in client_secret:
        print("❌ Invalid Spotify credentials")
        print("Please update spotify_credentials.py with your actual credentials")
        return
    
    print("✅ Spotify credentials loaded")
    print(f"Client ID: {client_id[:10]}...")
    print()
    
    # Initialize and run demo
    demo = WorkingEmotionMusicDemo(client_id, client_secret)
    demo.run_working_demo()

if __name__ == "__main__":
    main()