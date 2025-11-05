#!/usr/bin/env python3
"""
🎭🎵 Final High-Accuracy Emotion-Music Demo
Maximum accuracy emotion detection + Spotify integration
"""

import cv2
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import our advanced modules
from advanced_emotion_detection import get_advanced_emotion_detector, detect_emotion_advanced
from simple_spotify_auth import setup_spotify_simple, play_spotify_playlist

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalEmotionMusicDemo:
    """Final demo with maximum accuracy emotion detection and Spotify integration"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        
        # Initialize components
        self.face_cascade = self._setup_face_detection()
        self.emotion_detector = get_advanced_emotion_detector()
        self.spotify_client = None
        
        # Demo state
        self.current_emotion = None
        self.current_confidence = 0.0
        self.last_emotion_time = 0
        self.last_music_time = 0
        self.emotion_cooldown = 1.5  # Faster response
        self.music_cooldown = 8.0   # Prevent too frequent changes
        self.frame_count = 0
        self.is_spotify_connected = False
        
        # Enhanced playlist mapping
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
            'accuracy_score': 0.0,
            'session_start': datetime.now()
        }
        
        logger.info("✅ Final high-accuracy emotion-music demo initialized")
    
    def _setup_face_detection(self):
        """Setup optimized face detection"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                logger.error(f"❌ Haar cascade not found: {cascade_path}")
                return None
            
            face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("✅ Optimized face detection initialized")
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
            print("Setting up Spotify for music playback...")
            print("Browser will open for authentication.")
            print()
            
            self.spotify_client = setup_spotify_simple(self.client_id, self.client_secret)
            
            if self.spotify_client:
                self.is_spotify_connected = True
                logger.info("✅ Spotify connected successfully!")
                
                # Check for available devices
                try:
                    devices = self.spotify_client.devices()
                    if devices['devices']:
                        device_name = devices['devices'][0]['name']
                        logger.info(f"🎧 Ready to play on: {device_name}")
                    else:
                        logger.warning("⚠️ No active Spotify devices. Please open Spotify on a device.")
                        print("💡 Please open Spotify on your phone, computer, or web player")
                except:
                    pass
                
                return True
            else:
                logger.warning("⚠️ Spotify connection failed, running in demo mode")
                return False
                
        except Exception as e:
            logger.error(f"❌ Spotify setup error: {e}")
            return False
    
    def detect_faces_optimized(self, frame):
        """Optimized face detection for better accuracy"""
        if self.face_cascade is None:
            return []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhance image for better detection
            gray = cv2.equalizeHist(gray)
            
            # Multi-scale detection for better accuracy
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,  # Smaller steps for better detection
                minNeighbors=6,    # Higher threshold for more reliable detection
                minSize=(100, 100), # Larger minimum size for better accuracy
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Filter faces by size and position
            filtered_faces = []
            h, w = frame.shape[:2]
            
            for (x, y, fw, fh) in faces:
                # Ensure face is not too close to edges
                if (x > 20 and y > 20 and 
                    x + fw < w - 20 and y + fh < h - 20 and
                    fw > 100 and fh > 100):  # Minimum size for accuracy
                    filtered_faces.append((x, y, fw, fh))
            
            return filtered_faces
            
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
            
            # Detect emotion using advanced algorithm
            emotion, confidence = detect_emotion_advanced(frame, face_coords)
            
            # Update statistics
            self.stats['total_detections'] += 1
            self.stats['emotions_detected'][emotion] = self.stats['emotions_detected'].get(emotion, 0) + 1
            self.stats['accuracy_score'] = (self.stats['accuracy_score'] * (self.stats['total_detections'] - 1) + confidence) / self.stats['total_detections']
            
            # Only proceed if confidence is high enough
            if confidence > 0.65:  # Higher threshold for better accuracy
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
                            
                            # Visual feedback
                            print(f"\n🎵 NOW PLAYING: {playlist_name}")
                            print(f"🎭 Detected: {emotion.title()} {self.emotion_detector.get_emotion_emoji(emotion)} (Confidence: {confidence:.2f})")
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
    
    def draw_professional_overlay(self, frame, faces):
        """Draw professional-looking overlay with enhanced information"""
        try:
            h, w = frame.shape[:2]
            
            # Process faces
            for i, (x, y, face_w, face_h) in enumerate(faces):
                # Draw face rectangle with gradient effect
                cv2.rectangle(frame, (x-2, y-2), (x + face_w+2, y + face_h+2), (0, 255, 0), 4)
                cv2.rectangle(frame, (x, y), (x + face_w, y + face_h), (0, 200, 0), 2)
                
                # Process emotion
                emotion, confidence = self.process_emotion_and_music(frame, (x, y, face_w, face_h))
                
                if emotion:
                    emoji = self.emotion_detector.get_emotion_emoji(emotion)
                    
                    # Enhanced emotion display
                    emotion_text = f"{emotion.title()} {emoji}"
                    confidence_text = f"Accuracy: {confidence:.1%}"
                    
                    # Professional text background
                    text_y = y - 20
                    bg_height = 60
                    cv2.rectangle(frame, (x, text_y - bg_height), (x + face_w, text_y + 10), (0, 0, 0), -1)
                    cv2.rectangle(frame, (x, text_y - bg_height), (x + face_w, text_y + 10), (0, 255, 0), 2)
                    
                    # Draw emotion text with better styling
                    cv2.putText(frame, emotion_text, (x + 10, text_y - 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, confidence_text, (x + 10, text_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Show music info with enhanced styling
                    if emotion in self.playlist_names:
                        playlist_name = self.playlist_names[emotion]
                        music_text = f"♪ {playlist_name}"
                        
                        # Music info background
                        music_y = y + face_h + 15
                        cv2.rectangle(frame, (x, music_y), (x + face_w, music_y + 35), (0, 0, 0), -1)
                        cv2.rectangle(frame, (x, music_y), (x + face_w, music_y + 35), (255, 255, 0), 2)
                        
                        cv2.putText(frame, music_text, (x + 10, music_y + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Enhanced main info panel
            self._draw_professional_info_panel(frame, faces)
            
        except Exception as e:
            logger.error(f"❌ Error drawing overlay: {e}")
    
    def _draw_professional_info_panel(self, frame, faces):
        """Draw professional information panel"""
        try:
            h, w = frame.shape[:2]
            
            # Create gradient overlay
            overlay = frame.copy()
            panel_height = 160
            
            # Gradient background
            for i in range(panel_height):
                alpha = 0.9 - (i / panel_height) * 0.3
                cv2.rectangle(overlay, (10, 10 + i), (w - 10, 11 + i), (0, 0, 0), -1)
            
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Border
            cv2.rectangle(frame, (10, 10), (w - 10, panel_height + 10), (0, 255, 255), 3)
            
            # Title with enhanced styling
            title = "🎭🎵 FaceReco - High Accuracy Demo"
            cv2.putText(frame, title, (25, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
            cv2.putText(frame, title, (25, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            # Spotify status with icon
            spotify_status = "🎵 Spotify: Connected & Ready" if self.is_spotify_connected else "🎵 Spotify: Demo Mode"
            color = (0, 255, 0) if self.is_spotify_connected else (0, 255, 255)
            cv2.putText(frame, spotify_status, (25, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Current emotion with enhanced display
            if self.current_emotion:
                emotion_emoji = self.emotion_detector.get_emotion_emoji(self.current_emotion)
                current_text = f"Current Emotion: {self.current_emotion.title()} {emotion_emoji}"
                cv2.putText(frame, current_text, (25, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Confidence bar
                conf_width = int(200 * self.current_confidence)
                cv2.rectangle(frame, (25, 110), (225, 125), (100, 100, 100), -1)
                cv2.rectangle(frame, (25, 110), (25 + conf_width, 125), (0, 255, 0), -1)
                cv2.putText(frame, f"{self.current_confidence:.1%}", (235, 122), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Current playlist
                if self.current_emotion in self.playlist_names:
                    playlist_text = f"♪ Now Playing: {self.playlist_names[self.current_emotion]}"
                    cv2.putText(frame, playlist_text, (25, 145), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Statistics panel (right side)
            stats_x = w - 300
            cv2.putText(frame, f"Faces: {len(faces)}", (stats_x, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame, f"Detections: {self.stats['total_detections']}", (stats_x, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame, f"Avg Accuracy: {self.stats['accuracy_score']:.1%}", (stats_x, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Session time
            session_time = (datetime.now() - self.stats['session_start']).seconds
            cv2.putText(frame, f"Session: {session_time}s", (stats_x, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Controls (bottom)
            controls = [
                "🎮 Controls: ESC/Q=Quit | S=Save | P=Pause | R=Resume | SPACE=Reset",
                "🎭 Try different expressions: 😊 😢 😠 😐 😲"
            ]
            
            for i, text in enumerate(controls):
                cv2.putText(frame, text, (15, h - 35 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
        except Exception as e:
            logger.error(f"❌ Error drawing professional panel: {e}")
    
    def run_final_demo(self):
        """Run the final high-accuracy emotion-music demo"""
        logger.info("🎬 Starting Final High-Accuracy Emotion-Music Demo")
        logger.info("=" * 60)
        
        print("🎭🎵 FaceReco Final Demo")
        print("=" * 30)
        print("🎯 Maximum accuracy emotion detection")
        print("🎵 Spotify music integration")
        print("🎬 Professional real-time overlay")
        print()
        
        # Setup Spotify
        spotify_setup = input("Setup Spotify for music playback? (y/n): ").lower().strip()
        if spotify_setup == 'y':
            if not self.setup_spotify():
                print("⚠️ Continuing in demo mode...")
        else:
            logger.info("🎵 Running in demo mode (no Spotify)")
        
        # Initialize camera
        cap = self._initialize_camera_optimized()
        if cap is None:
            return False
        
        logger.info("🎭 Final demo started with maximum accuracy!")
        logger.info("📋 Try different expressions to see high-accuracy detection")
        if self.is_spotify_connected:
            logger.info("🎵 Music will play automatically based on your emotions!")
        
        print("\n🎬 Demo is running! Look at the camera and try different expressions:")
        print("😊 Smile widely for happy music")
        print("😢 Frown for chill music") 
        print("😠 Make an angry face for energetic music")
        print("😐 Stay neutral for focus music")
        print("😲 Look surprised for surprise music")
        print()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("❌ Failed to read frame")
                    break
                
                self.frame_count += 1
                
                # Detect faces with optimization
                faces = self.detect_faces_optimized(frame)
                
                # Draw professional overlay
                self.draw_professional_overlay(frame, faces)
                
                # Display frame
                cv2.imshow('FaceReco - Final High-Accuracy Demo', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("🛑 Stopping final demo...")
                    break
                elif key == ord('s'):  # Save frame
                    filename = f'final_demo_{int(time.time())}.jpg'
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
                    self.emotion_detector.reset_history()
                    logger.info("🔄 Reset emotion detection and history")
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Demo interrupted by user")
        
        except Exception as e:
            logger.error(f"❌ Demo error: {e}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_final_stats()
            logger.info("✅ Final demo completed")
        
        return True
    
    def _initialize_camera_optimized(self):
        """Initialize camera with optimal settings for accuracy"""
        try:
            # Try AVFoundation backend first (macOS)
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                logger.error("❌ Could not open camera")
                return None
            
            # Set optimal camera properties for accuracy
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)   # Higher resolution
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)     # Optimal brightness
            cap.set(cv2.CAP_PROP_CONTRAST, 0.6)       # Good contrast
            cap.set(cv2.CAP_PROP_SATURATION, 0.5)     # Balanced saturation
            
            logger.info("✅ Camera initialized with optimal settings for maximum accuracy")
            return cap
            
        except Exception as e:
            logger.error(f"❌ Error initializing camera: {e}")
            return None
    
    def _print_final_stats(self):
        """Print comprehensive final statistics"""
        try:
            duration = (datetime.now() - self.stats['session_start']).seconds
            
            print("\n📊 Final Demo Session Statistics")
            print("=" * 50)
            print(f"Duration: {duration}s")
            print(f"Frames processed: {self.frame_count}")
            print(f"Emotion detections: {self.stats['total_detections']}")
            print(f"Average accuracy: {self.stats['accuracy_score']:.1%}")
            print(f"Spotify status: {'Connected ✅' if self.is_spotify_connected else 'Demo Mode ⚠️'}")
            
            if self.stats['emotions_detected']:
                print("\n🎭 Emotions detected:")
                for emotion, count in self.stats['emotions_detected'].items():
                    emoji = self.emotion_detector.get_emotion_emoji(emotion)
                    percentage = (count / self.stats['total_detections']) * 100
                    print(f"  {emotion.title()} {emoji}: {count} ({percentage:.1f}%)")
            
            if self.stats['playlists_played']:
                print("\n🎵 Playlists played:")
                for emotion, count in self.stats['playlists_played'].items():
                    playlist_name = self.playlist_names[emotion]
                    print(f"  {playlist_name}: {count} times")
            
            print(f"\n🎯 Performance Summary:")
            print(f"  Detection Rate: {self.stats['total_detections'] / max(1, duration):.1f} detections/second")
            print(f"  Accuracy Score: {self.stats['accuracy_score']:.1%}")
            print(f"  Music Integration: {'✅ Active' if self.is_spotify_connected else '⚠️ Demo Mode'}")
            
        except Exception as e:
            logger.error(f"❌ Error printing final stats: {e}")

def main():
    """Main function"""
    print("🎭🎵 FaceReco Final High-Accuracy Emotion-Music Demo")
    print("=" * 60)
    print("🎯 Maximum Accuracy Emotion Detection")
    print("🎵 Full Spotify Music Integration")
    print("🎬 Professional Real-time Interface")
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
    
    # Initialize and run final demo
    demo = FinalEmotionMusicDemo(client_id, client_secret)
    demo.run_final_demo()

if __name__ == "__main__":
    main()