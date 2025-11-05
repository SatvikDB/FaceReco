#!/usr/bin/env python3
"""
🎵 Spotify Integration Module for FaceReco
Handles Spotify authentication and playlist playback based on emotions
"""

import os
import logging
import json
from typing import Dict, Optional, List
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyController:
    """
    Spotify API controller for emotion-based music playback
    """
    
    def __init__(self, client_id: str = None, client_secret: str = None, redirect_uri: str = None):
        """
        Initialize Spotify controller
        
        Args:
            client_id (str): Spotify Client ID
            client_secret (str): Spotify Client Secret  
            redirect_uri (str): Spotify Redirect URI
        """
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        self.redirect_uri = redirect_uri or os.getenv('SPOTIFY_REDIRECT_URI', 'http://127.0.0.1:8888/callback')
        
        self.sp = None
        self.is_authenticated = False
        self.current_device_id = None
        
        # Emotion to playlist mapping
        self.emotion_to_playlist = {
            "happy": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
            "sad": "https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1", 
            "neutral": "https://open.spotify.com/playlist/37i9dQZF1DWXRqgorJj26U",
            "angry": "https://open.spotify.com/playlist/37i9dQZF1DWYxwmBaMqxsl"
        }
        
        # Playlist names for display
        self.playlist_names = {
            "happy": "Happy Vibes 🎉",
            "sad": "Chill & Relax 🌙", 
            "neutral": "Focus Flow 🎧",
            "angry": "Energy Boost ⚡"
        }
        
        # Initialize Spotify client
        self._initialize_spotify()
    
    def _initialize_spotify(self):
        """Initialize Spotify client with authentication"""
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyOAuth
            
            if not all([self.client_id, self.client_secret]):
                logger.warning("⚠️ Spotify credentials not provided. Using demo mode.")
                self._setup_demo_mode()
                return
            
            # Set up OAuth
            scope = "user-read-playback-state user-modify-playback-state playlist-read-private user-read-currently-playing"
            
            auth_manager = SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope=scope,
                cache_path=".spotify_cache"
            )
            
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            self.is_authenticated = True
            
            # Get current user info
            user_info = self.sp.current_user()
            logger.info(f"✅ Authenticated as: {user_info['display_name']}")
            
            # Get available devices
            self._get_available_devices()
            
        except ImportError:
            logger.warning("❌ Spotipy not installed. Installing...")
            self._install_spotipy()
        except Exception as e:
            logger.error(f"❌ Spotify authentication failed: {e}")
            self._setup_demo_mode()
    
    def _install_spotipy(self):
        """Install spotipy library"""
        try:
            import subprocess
            import sys
            
            logger.info("📦 Installing spotipy...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "spotipy"])
            
            # Try to initialize again
            self._initialize_spotify()
            
        except Exception as e:
            logger.error(f"❌ Failed to install spotipy: {e}")
            self._setup_demo_mode()
    
    def _setup_demo_mode(self):
        """Setup demo mode when Spotify is not available"""
        logger.info("🎵 Running in demo mode - no actual Spotify playback")
        self.is_authenticated = False
    
    def _get_available_devices(self):
        """Get available Spotify devices"""
        try:
            if not self.sp:
                return
            
            devices = self.sp.devices()
            active_devices = [d for d in devices['devices'] if d['is_active']]
            
            if active_devices:
                self.current_device_id = active_devices[0]['id']
                logger.info(f"🎧 Active device: {active_devices[0]['name']}")
            elif devices['devices']:
                self.current_device_id = devices['devices'][0]['id']
                logger.info(f"📱 Available device: {devices['devices'][0]['name']}")
            else:
                logger.warning("⚠️ No Spotify devices found. Please open Spotify on a device.")
                
        except Exception as e:
            logger.error(f"❌ Error getting devices: {e}")
    
    def play_playlist_for_emotion(self, emotion: str) -> bool:
        """
        Play Spotify playlist based on detected emotion
        
        Args:
            emotion (str): Detected emotion
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_authenticated:
                logger.info(f"🎵 [DEMO] Would play {self.playlist_names.get(emotion, 'Unknown')} playlist for emotion: {emotion}")
                return True
            
            playlist_url = self.emotion_to_playlist.get(emotion)
            if not playlist_url:
                logger.warning(f"⚠️ No playlist found for emotion: {emotion}")
                return False
            
            # Extract playlist ID from URL
            playlist_id = self._extract_playlist_id(playlist_url)
            if not playlist_id:
                logger.error(f"❌ Invalid playlist URL: {playlist_url}")
                return False
            
            # Start playback
            playlist_uri = f"spotify:playlist:{playlist_id}"
            
            try:
                self.sp.start_playback(
                    device_id=self.current_device_id,
                    context_uri=playlist_uri
                )
                
                playlist_name = self.playlist_names.get(emotion, f"{emotion.title()} Playlist")
                logger.info(f"🎵 Now playing: {playlist_name}")
                return True
                
            except Exception as playback_error:
                # Try without device_id if it fails
                if "NO_ACTIVE_DEVICE" in str(playback_error):
                    logger.warning("⚠️ No active device, trying to transfer playback...")
                    self._transfer_playback()
                    return self.play_playlist_for_emotion(emotion)
                else:
                    logger.error(f"❌ Playback error: {playback_error}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error playing playlist for emotion {emotion}: {e}")
            return False
    
    def _extract_playlist_id(self, playlist_url: str) -> Optional[str]:
        """Extract playlist ID from Spotify URL"""
        try:
            # Handle different URL formats
            if "playlist/" in playlist_url:
                playlist_id = playlist_url.split("playlist/")[-1].split("?")[0]
                return playlist_id
            return None
        except Exception as e:
            logger.error(f"❌ Error extracting playlist ID: {e}")
            return None
    
    def _transfer_playback(self):
        """Transfer playback to available device"""
        try:
            devices = self.sp.devices()
            if devices['devices']:
                device_id = devices['devices'][0]['id']
                self.sp.transfer_playback(device_id=device_id, force_play=False)
                self.current_device_id = device_id
                logger.info(f"🔄 Transferred playback to: {devices['devices'][0]['name']}")
        except Exception as e:
            logger.error(f"❌ Error transferring playback: {e}")
    
    def get_current_playback(self) -> Optional[Dict]:
        """Get current playback information"""
        try:
            if not self.sp:
                return None
            
            current = self.sp.current_playback()
            if current and current['is_playing']:
                track = current['item']
                return {
                    'track_name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'is_playing': current['is_playing']
                }
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting current playback: {e}")
            return None
    
    def pause_playback(self):
        """Pause current playback"""
        try:
            if self.sp:
                self.sp.pause_playback()
                logger.info("⏸️ Playback paused")
        except Exception as e:
            logger.error(f"❌ Error pausing playback: {e}")
    
    def resume_playback(self):
        """Resume playback"""
        try:
            if self.sp:
                self.sp.start_playback()
                logger.info("▶️ Playback resumed")
        except Exception as e:
            logger.error(f"❌ Error resuming playback: {e}")
    
    def get_playlist_name_for_emotion(self, emotion: str) -> str:
        """Get display name for emotion playlist"""
        return self.playlist_names.get(emotion, f"{emotion.title()} Playlist")
    
    def update_playlist_mapping(self, emotion: str, playlist_url: str, playlist_name: str = None):
        """Update emotion to playlist mapping"""
        self.emotion_to_playlist[emotion] = playlist_url
        if playlist_name:
            self.playlist_names[emotion] = playlist_name
        logger.info(f"✅ Updated playlist for {emotion}: {playlist_name or playlist_url}")
    
    def save_config(self, config_path: str = "spotify_config.json"):
        """Save current configuration to file"""
        try:
            config = {
                'emotion_to_playlist': self.emotion_to_playlist,
                'playlist_names': self.playlist_names,
                'redirect_uri': self.redirect_uri
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"💾 Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving config: {e}")
    
    def load_config(self, config_path: str = "spotify_config.json"):
        """Load configuration from file"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                self.emotion_to_playlist.update(config.get('emotion_to_playlist', {}))
                self.playlist_names.update(config.get('playlist_names', {}))
                
                logger.info(f"📂 Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")

# Global Spotify controller instance
_spotify_controller = None

def get_spotify_controller(client_id: str = None, client_secret: str = None, redirect_uri: str = None) -> SpotifyController:
    """Get global Spotify controller instance"""
    global _spotify_controller
    if _spotify_controller is None:
        _spotify_controller = SpotifyController(client_id, client_secret, redirect_uri)
    return _spotify_controller

def play_music_for_emotion(emotion: str) -> bool:
    """
    Convenience function to play music for emotion
    
    Args:
        emotion (str): Detected emotion
        
    Returns:
        bool: True if successful
    """
    controller = get_spotify_controller()
    return controller.play_playlist_for_emotion(emotion)

def get_playlist_name_for_emotion(emotion: str) -> str:
    """Get playlist name for emotion"""
    controller = get_spotify_controller()
    return controller.get_playlist_name_for_emotion(emotion)

# Logging function for emotion-music events
def log_emotion_music_event(user_name: str, emotion: str, confidence: float, playlist_name: str):
    """Log emotion detection and music playback event"""
    try:
        log_file = "emotion_music_log.csv"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create header if file doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write("Timestamp,User,Emotion,Confidence,Playlist,Status\n")
        
        # Append log entry
        with open(log_file, 'a') as f:
            f.write(f"{timestamp},{user_name},{emotion},{confidence:.2f},{playlist_name},Success\n")
        
        logger.info(f"📝 Logged emotion-music event: {user_name} -> {emotion} -> {playlist_name}")
        
    except Exception as e:
        logger.error(f"❌ Error logging event: {e}")

if __name__ == "__main__":
    # Test Spotify integration
    print("🧪 Testing Spotify Integration Module")
    print("=" * 40)
    
    # Test with demo mode (no credentials)
    controller = SpotifyController()
    
    print(f"🔐 Authentication status: {controller.is_authenticated}")
    
    # Test emotion playlist mapping
    test_emotions = ["happy", "sad", "neutral", "angry"]
    
    for emotion in test_emotions:
        playlist_name = controller.get_playlist_name_for_emotion(emotion)
        print(f"🎵 {emotion.title()}: {playlist_name}")
        
        # Test playback (demo mode)
        success = controller.play_playlist_for_emotion(emotion)
        print(f"   Playback: {'✅ Success' if success else '❌ Failed'}")
    
    # Test logging
    log_emotion_music_event("TestUser", "happy", 0.85, "Happy Vibes 🎉")
    
    print("\n✅ Spotify integration module test completed")
    print("\n💡 To use with real Spotify:")
    print("   1. Set environment variables:")
    print("      export SPOTIFY_CLIENT_ID='your_client_id'")
    print("      export SPOTIFY_CLIENT_SECRET='your_client_secret'")
    print("   2. Or pass credentials to SpotifyController()")
    print("   3. Make sure Spotify is open on a device")