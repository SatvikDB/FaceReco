#!/usr/bin/env python3
"""
🎵 Simple Spotify Authentication
Simplified approach for Spotify integration
"""

import os
import webbrowser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_spotify_simple(client_id: str, client_secret: str):
    """Simple Spotify setup with manual token input"""
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
        
        # Setup OAuth with prompt for manual input
        scope = "user-read-playback-state user-modify-playback-state playlist-read-private user-read-currently-playing"
        
        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri="http://127.0.0.1:8888/callback",
            scope=scope,
            cache_path=".spotify_cache_simple",
            open_browser=True
        )
        
        # Get Spotify client
        sp = spotipy.Spotify(auth_manager=auth_manager)
        
        # Test connection
        user_info = sp.current_user()
        logger.info(f"✅ Connected to Spotify as: {user_info['display_name']}")
        
        return sp
        
    except Exception as e:
        logger.error(f"❌ Spotify authentication failed: {e}")
        return None

def play_spotify_playlist(sp, emotion: str):
    """Play Spotify playlist for emotion"""
    try:
        if not sp:
            return False
        
        # Playlist mapping
        playlists = {
            "happy": "37i9dQZF1DXdPec7aLTmlC",
            "sad": "37i9dQZF1DX7qK8ma5wgG1", 
            "neutral": "37i9dQZF1DWXRqgorJj26U",
            "angry": "37i9dQZF1DWYxwmBaMqxsl",
            "surprised": "37i9dQZF1DXdPec7aLTmlC"
        }
        
        playlist_id = playlists.get(emotion)
        if not playlist_id:
            return False
        
        # Get available devices
        devices = sp.devices()
        if not devices['devices']:
            logger.warning("⚠️ No Spotify devices available. Please open Spotify on a device.")
            return False
        
        # Use first available device
        device_id = devices['devices'][0]['id']
        
        # Start playback
        playlist_uri = f"spotify:playlist:{playlist_id}"
        sp.start_playback(
            device_id=device_id,
            context_uri=playlist_uri
        )
        
        logger.info(f"🎵 Playing {emotion} playlist on {devices['devices'][0]['name']}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error playing playlist: {e}")
        return False

if __name__ == "__main__":
    # Test simple Spotify auth
    from spotify_credentials import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
    
    print("🧪 Testing Simple Spotify Authentication")
    sp = setup_spotify_simple(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    
    if sp:
        print("✅ Spotify authentication successful!")
        print("🎵 Testing playlist playback...")
        success = play_spotify_playlist(sp, "happy")
        print(f"Playlist test: {'✅ Success' if success else '❌ Failed'}")
    else:
        print("❌ Spotify authentication failed")