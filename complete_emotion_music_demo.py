#!/usr/bin/env python3
"""
🎭🎵 Complete FaceReco Emotion-Music Demo
Advanced emotion detection + Full Spotify integration
"""

import cv2
import os
import sys
import time
import logging
import webbrowser
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import our modules
from improved_emotion_detection import get_improved_emotion_detector, detect_emotion_improved
from spotify_oauth_server import SpotifyOAuthServer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteEmotionMusicDemo:
    """Complete demo with advanced emotion detection and Spotify integration"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = "http://127.0.0.1:8888/callback"
        
        # Initialize components
        self.face_cascade = self._setup_face_detection()
        self.emotion_detector = get_improved_emotion_detector()
        self.spotify_controller = None
        self.oauth_server = None
        
        # Demo state
        self.current_emotion = None
        self.current_confidence = 0.0
        self.last_emotion_time = 0
        self.emotion_cooldown = 2.0  # Reduced cooldown for better responsiveness
        self.frame_count = 0
        self.is_spotify_authenticated = False
        
        # Playlist mapping
        self.emotion_playlists = {
            "happy": "37i9dQZF1DXdPec7aLTmlC",
            "sad": "37i9dQZF1DX7qK8ma5wgG1", 
            "neutral": "37i9dQZF1DWXRqgorJj26U",
            "angry": "37i9dQZF1DWYxwmBaMqxsl",
            "surprised": "37i9dQZF1DXdPec7aLTmlC"  # Use happy playlist for surprised
        }
        
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
        
        logger.info("✅ Complete emotion-music demo initialized")
    
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
    
    def setup_spotify_authentication(self):
        """Setup Spotify authentication with OAuth server"""
        try:
            logger.info("🎵 Setting up Spotify authentication...")
            
            # Start OAuth server
            self.oauth_server = SpotifyOAuthServer(port=8888)
            if not self.oauth_server.start_server():
                logger.error("❌ Failed to start OAuth server")
                return False
            
            # Import spotipy
            try:
                import spotipy
                from spotipy.oauth2 import SpotifyOAuth
            except ImportError:
                logger.error("❌ Spotipy not installed")
                return False
            
            # Setup OAuth
            scope = "user-read-playback-state user-modify-playback-state playlist-read-private user-read-currently-playing"
            
            auth_manager = SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope=scope,
                cache_path=".spotify_cache_complete"
            )
            
            # Get authorization URL
            auth_url = auth_manager.get_authorize_url()
            
            logger.info("🌐 Opening Spotify authorization in browser...")
            webbrowser.open(auth_url)
            
            print("\n🎵 Spotify Authentication")
            print("=" * 30)
            print("1. Browser opened for Spotify login")
            print("2. Log in and authorize FaceReco")
            print("3. Waiting for authorization...")
            
            # Wait for callback
            if self.oauth_server.wait_for_callback(timeout=120):
                logger.info("✅ Authorization received!")
                
                # Create Spotify client
                self.spotify_controller = spotipy.Spotify(auth_manager=auth_manager)
                
                # Test connection
                user_info = self.spotify_controller.current_user()
                logger.info(f"✅ Connected as: {user_info['display_name']}")
                
                self.is_spotify_authenticated = True
                
                # Stop OAuth server
                self.oauth_server.stop_server()
                
                return True
            else:
                logger.error("❌ Spotify authorization failed or timed out")
                self.oauth_server.stop_server()
                return False
                
        except Exception as e:
            logger.error(f"❌ Spotify authentication error: {e}")
            if self.oauth_server:
                self.oauth_server.stop_server()
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
                minSize=(80, 80)  # Larger minimum size for better detection
            )
            return faces
            
        except Exception as e:
            logger.error(f"❌ Error detecting faces: {e}")
            return []
    
    def process_emotion_and_music(self, frame, face_coords):
        """Process emotion detection and trigger music playback"""
        try:
            current_time = time.time()
            
            # Check cooldown
            if current_time - self.last_emotion_time < self.emotion_cooldown:
                return self.current_emotion, self.current_confidence
            
            # Detect emotion using improved algorithm
            emotion, confidence = detect_emotion_improved(frame, face_coords)
            
            # Update statistics
            self.stats['total_detections'] += 1
            self.stats['emotions_detected'][emotion] = self.stats['emotions_detected'].get(emotion, 0) + 1
            
            # Only proceed if confidence is high enough and emotion changed
            if confidence > 0.6 and emotion != self.current_emotion:
                self.current_emotion = emotion
                self.current_confidence = confidence
                self.last_emotion_time = current_time
                
                # Play music for emotion
                if self.is_spotify_authenticated:
                    success = self._play_spotify_playlist(emotion)
                    if success:
                        self.stats['playlists_played'][emotion] = self.stats['playlists_played'].get(emotion, 0) + 1
                        playlist_name = self.playlist_names.get(emotion, f"{emotion.title()} Playlist")
                        logger.info(f"🎭🎵 {emotion.title()} ({confidence:.2f}) | Playing: {playlist_name}")
                else:
                    # Demo mode
                    playlist_name = self.playlist_names.get(emotion, f"{emotion.title()} Playlist")
                    logger.info(f"🎭🎵 [DEMO] {emotion.title()} ({confidence:.2f}) | Would play: {playlist_name}")
            
            return self.current_emotion, self.current_confidence
            
        except Exception as e:
            logger.error(f"❌ Error processing emotion and music: {e}")
            return "neutral", 0.5
    
    def _play_spotify_playlist(self, emotion: str) -> bool:
        """Play Spotify playlist for emotion"""
        try:
            if not self.spotify_controller:
                return False
            
            playlist_id = self.emotion_playlists.get(emotion)
            if not playlist_id:
                logger.warning(f"⚠️ No playlist for emotion: {emotion}")
                return False
            
            # Get available devices
            devices = self.spotify_controller.devices()
            if not devices['devices']:
                logger.warning("⚠️ No Spotify devices available")
                return False
            
            # Use first available device
            device_id = devices['devices'][0]['id']
            
            # Start playback
            playlist_uri = f"spotify:playlist:{playlist_id}"
            self.spotify_controller.start_playback(
                device_id=device_id,
                context_uri=playlist_uri
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error playing Spotify playlist: {e}")
            return False
    
    def draw_enhanced_overlay(self, frame, faces):
        """Draw enhanced information overlay"""
        try:
            h, w = frame.shape[:2]
            
            # Process faces
            for i, (x, y, face_w, face_h) in enumerate(faces):
                # Draw face rectangle with thicker border
                cv2.rectangle(frame, (x, y), (x + face_w, y + face_h), (0, 255, 0), 3)
                
                # Process emotion
                emotion, confidence = self.process_emotion_and_music(frame, (x, y, face_w, face_h))
                
                if emotion:
                    emoji = self.emotion_detector.get_emotion_emoji(emotion)
                    
                    # Draw emotion info with better styling
                    emotion_text = f"{emotion.title()} {emoji}"
                    confidence_text = f"Confidence: {confidence:.2f}"
                    
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
            
            # Enhanced main overlay
            self._draw_main_info_panel(frame, faces)
            
        except Exception as e:
            logger.error(f"❌ Error drawing overlay: {e}")
    
    def _draw_main_info_panel(self, frame, faces):
        """Draw main information panel"""
        try:
            h, w = frame.shape[:2]
            
            # Create semi-transparent overlay
            overlay = frame.copy()
            panel_height = 140
            cv2.rectangle(overlay, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Title
            title = "🎭🎵 FaceReco Complete Demo"
            cv2.putText(frame, title, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Spotify status
            spotify_status = "🎵 Spotify: Connected" if self.is_spotify_authenticated else "🎵 Spotify: Demo Mode"
            cv2.putText(frame, spotify_status, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Current emotion
            if self.current_emotion:
                emotion_emoji = self.emotion_detector.get_emotion_emoji(self.current_emotion)
                current_text = f"Current: {self.current_emotion.title()} {emotion_emoji}"
                cv2.putText(frame, current_text, (20, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Current playlist
                if self.current_emotion in self.playlist_names:
                    playlist_text = f"♪ {self.playlist_names[self.current_emotion]}"
                    cv2.putText(frame, playlist_text, (20, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Face count
            face_text = f"Faces: {len(faces)}"
            cv2.putText(frame, face_text, (20, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Statistics (bottom right)
            stats_text = [
                f"Detections: {self.stats['total_detections']}",
                f"Session: {(datetime.now() - self.stats['session_start']).seconds}s"
            ]
            
            for i, text in enumerate(stats_text):
                cv2.putText(frame, text, (w - 250, h - 40 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Controls (bottom left)
            controls = [
                "Controls:",
                "ESC/Q: Quit",
                "S: Save frame", 
                "P: Pause music",
                "R: Resume music",
                "SPACE: Reset"
            ]
            
            for i, text in enumerate(controls):
                cv2.putText(frame, text, (10, h - 120 + i * 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
        except Exception as e:
            logger.error(f"❌ Error drawing main panel: {e}")
    
    def run_complete_demo(self):
        """Run the complete emotion-music demo"""
        logger.info("🎬 Starting Complete Emotion-Music Demo")
        logger.info("=" * 50)
        
        # Setup Spotify authentication
        if not self.setup_spotify_authentication():
            logger.warning("⚠️ Spotify authentication failed, running in demo mode")
            self.is_spotify_authenticated = False
        
        # Initialize camera
        cap = self._initialize_camera()
        if cap is None:
            return False
        
        logger.info("🎭 Complete demo started!")
        logger.info("📋 Try different expressions to see improved emotion detection")
        
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
                cv2.imshow('FaceReco - Complete Emotion Music Demo', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("🛑 Stopping demo...")
                    break
                elif key == ord('s'):  # Save frame
                    filename = f'complete_demo_{int(time.time())}.jpg'
                    cv2.imwrite(filename, frame)
                    logger.info(f"📸 Saved: {filename}")
                elif key == ord('p'):  # Pause music
                    if self.is_spotify_authenticated and self.spotify_controller:
                        try:
                            self.spotify_controller.pause_playback()
                            logger.info("⏸️ Music paused")
                        except:
                            pass
                elif key == ord('r'):  # Resume music
                    if self.is_spotify_authenticated and self.spotify_controller:
                        try:
                            self.spotify_controller.start_playback()
                            logger.info("▶️ Music resumed")
                        except:
                            pass
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
            if self.oauth_server:
                self.oauth_server.stop_server()
            self._print_session_stats()
            logger.info("✅ Complete demo finished")
        
        return True
    
    def _initialize_camera(self):
        """Initialize camera with optimal settings"""
        try:
            # Try AVFoundation backend first (macOS)
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                logger.error("❌ Could not open camera")
                return None
            
            # Set optimal camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
            cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
            
            logger.info("✅ Camera initialized with optimal settings")
            return cap
            
        except Exception as e:
            logger.error(f"❌ Error initializing camera: {e}")
            return None
    
    def _print_session_stats(self):
        """Print comprehensive session statistics"""
        try:
            duration = (datetime.now() - self.stats['session_start']).seconds
            
            print("\n📊 Complete Demo Session Statistics")
            print("=" * 40)
            print(f"Duration: {duration}s")
            print(f"Frames processed: {self.frame_count}")
            print(f"Emotion detections: {self.stats['total_detections']}")
            print(f"Spotify status: {'Connected' if self.is_spotify_authenticated else 'Demo Mode'}")
            
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
    print("🎭🎵 FaceReco Complete Emotion-Music Demo")
    print("=" * 50)
    print("Advanced Emotion Detection + Full Spotify Integration")
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
    
    # Initialize and run demo
    demo = CompleteEmotionMusicDemo(client_id, client_secret)
    demo.run_complete_demo()

if __name__ == "__main__":
    main()