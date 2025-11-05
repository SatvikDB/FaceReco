#!/usr/bin/env python3
"""
🎵 Spotify Configuration Setup for FaceReco
Easy setup script for Spotify API credentials and playlist configuration
"""

import os
import json
import webbrowser
from typing import Dict

def setup_spotify_credentials():
    """Interactive setup for Spotify credentials"""
    print("🎵 Spotify API Setup")
    print("=" * 30)
    
    print("To use Spotify integration, you need to create a Spotify App:")
    print("1. Go to https://developer.spotify.com/dashboard")
    print("2. Log in with your Spotify account")
    print("3. Click 'Create an App'")
    print("4. Fill in app details:")
    print("   - App name: 'FaceReco Emotion Music'")
    print("   - App description: 'Emotion-based music recommendation'")
    print("5. In app settings, add redirect URI: https://localhost:8888/callback")
    print()
    
    # Ask if user wants to open the dashboard
    open_dashboard = input("Open Spotify Developer Dashboard? (y/n): ").lower().strip()
    if open_dashboard == 'y':
        try:
            webbrowser.open("https://developer.spotify.com/dashboard")
            print("✅ Opened Spotify Developer Dashboard")
        except:
            print("❌ Could not open browser")
    
    print("\nAfter creating your app, enter the credentials:")
    
    # Get credentials from user
    client_id = input("Enter your Spotify Client ID: ").strip()
    client_secret = input("Enter your Spotify Client Secret: ").strip()
    redirect_uri = input("Enter Redirect URI (default: https://localhost:8888/callback): ").strip()
    
    if not redirect_uri:
        redirect_uri = "https://localhost:8888/callback"
    
    if not client_id or not client_secret:
        print("❌ Client ID and Client Secret are required")
        return None
    
    return {
        'client_id': client_id,
        'client_secret': client_secret,
        'redirect_uri': redirect_uri
    }

def setup_playlist_configuration():
    """Setup emotion to playlist mapping"""
    print("\n🎶 Playlist Configuration")
    print("=" * 30)
    
    print("Configure playlists for each emotion:")
    print("You can use Spotify playlist URLs or leave default for Spotify's curated playlists")
    print()
    
    # Default playlists (Spotify's curated playlists)
    default_playlists = {
        "happy": {
            "url": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
            "name": "Happy Hits! 🎉"
        },
        "sad": {
            "url": "https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1",
            "name": "Sad Songs 😢"
        },
        "neutral": {
            "url": "https://open.spotify.com/playlist/37i9dQZF1DWXRqgorJj26U",
            "name": "Chill Hits 😌"
        },
        "angry": {
            "url": "https://open.spotify.com/playlist/37i9dQZF1DWYxwmBaMqxsl",
            "name": "Beast Mode 😤"
        }
    }
    
    emotion_playlists = {}
    playlist_names = {}
    
    for emotion, default in default_playlists.items():
        print(f"\n{emotion.title()} Emotion:")
        print(f"Default: {default['name']}")
        
        use_custom = input(f"Use custom playlist for {emotion}? (y/n): ").lower().strip()
        
        if use_custom == 'y':
            custom_url = input(f"Enter Spotify playlist URL for {emotion}: ").strip()
            custom_name = input(f"Enter display name for {emotion} playlist: ").strip()
            
            if custom_url:
                emotion_playlists[emotion] = custom_url
                playlist_names[emotion] = custom_name or f"{emotion.title()} Playlist"
            else:
                emotion_playlists[emotion] = default['url']
                playlist_names[emotion] = default['name']
        else:
            emotion_playlists[emotion] = default['url']
            playlist_names[emotion] = default['name']
    
    return emotion_playlists, playlist_names

def save_configuration(credentials: Dict, playlists: Dict, playlist_names: Dict):
    """Save configuration to files"""
    try:
        # Save Spotify credentials to environment file
        env_content = f"""# Spotify API Credentials for FaceReco
export SPOTIFY_CLIENT_ID="{credentials['client_id']}"
export SPOTIFY_CLIENT_SECRET="{credentials['client_secret']}"
export SPOTIFY_REDIRECT_URI="{credentials['redirect_uri']}"

# To use these credentials, run:
# source spotify_env.sh
"""
        
        with open("spotify_env.sh", "w") as f:
            f.write(env_content)
        
        print("✅ Saved credentials to spotify_env.sh")
        
        # Save playlist configuration
        config = {
            'emotion_to_playlist': playlists,
            'playlist_names': playlist_names,
            'redirect_uri': credentials['redirect_uri']
        }
        
        with open("spotify_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("✅ Saved playlist configuration to spotify_config.json")
        
        # Create a Python config file for easy import
        python_config = f'''"""
Spotify Configuration for FaceReco
Auto-generated configuration file
"""

# Spotify API Credentials
SPOTIFY_CLIENT_ID = "{credentials['client_id']}"
SPOTIFY_CLIENT_SECRET = "{credentials['client_secret']}"
SPOTIFY_REDIRECT_URI = "{credentials['redirect_uri']}"

# Emotion to Playlist Mapping
EMOTION_PLAYLISTS = {repr(playlists)}

# Playlist Display Names
PLAYLIST_NAMES = {repr(playlist_names)}
'''
        
        with open("spotify_credentials.py", "w") as f:
            f.write(python_config)
        
        print("✅ Saved Python configuration to spotify_credentials.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

def test_spotify_connection(credentials: Dict):
    """Test Spotify connection"""
    print("\n🧪 Testing Spotify Connection")
    print("=" * 30)
    
    try:
        # Set environment variables temporarily
        os.environ['SPOTIFY_CLIENT_ID'] = credentials['client_id']
        os.environ['SPOTIFY_CLIENT_SECRET'] = credentials['client_secret']
        os.environ['SPOTIFY_REDIRECT_URI'] = credentials['redirect_uri']
        
        # Import and test Spotify controller
        from spotify_integration import SpotifyController
        
        controller = SpotifyController()
        
        if controller.is_authenticated:
            print("✅ Spotify authentication successful!")
            
            # Test getting current user
            if controller.sp:
                user_info = controller.sp.current_user()
                print(f"👤 Logged in as: {user_info.get('display_name', 'Unknown')}")
            
            # Test playlist access
            print("🎵 Testing playlist access...")
            for emotion in ['happy', 'sad', 'neutral', 'angry']:
                playlist_name = controller.get_playlist_name_for_emotion(emotion)
                print(f"  {emotion.title()}: {playlist_name}")
            
            return True
        else:
            print("⚠️ Running in demo mode - authentication not available")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def main():
    """Main configuration setup"""
    print("🎭🎵 FaceReco Spotify Configuration Setup")
    print("=" * 50)
    
    # Setup credentials
    credentials = setup_spotify_credentials()
    if not credentials:
        print("❌ Setup cancelled")
        return
    
    # Setup playlists
    playlists, playlist_names = setup_playlist_configuration()
    
    # Save configuration
    if save_configuration(credentials, playlists, playlist_names):
        print("\n✅ Configuration saved successfully!")
        
        # Test connection
        test_connection = input("\nTest Spotify connection now? (y/n): ").lower().strip()
        if test_connection == 'y':
            test_spotify_connection(credentials)
        
        print("\n🚀 Setup Complete!")
        print("=" * 20)
        print("To use the emotion-music demo:")
        print("1. Run: source spotify_env.sh")
        print("2. Run: python3 emotion_music_demo.py")
        print()
        print("Or import credentials in Python:")
        print("from spotify_credentials import *")
        
    else:
        print("❌ Configuration setup failed")

if __name__ == "__main__":
    main()