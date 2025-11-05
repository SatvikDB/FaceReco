"""
Spotify Configuration for FaceReco
Auto-generated configuration file
"""

# Spotify API Credentials
SPOTIFY_CLIENT_ID = "79d78f5a437c4258b8a95d3a7c68601a"
SPOTIFY_CLIENT_SECRET = "382bd6e8a3f54e2abbaa297eefba0e53"
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8888/callback"

# Emotion to Playlist Mapping
EMOTION_PLAYLISTS = {
    "happy": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
    "sad": "https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1", 
    "neutral": "https://open.spotify.com/playlist/37i9dQZF1DWXRqgorJj26U",
    "angry": "https://open.spotify.com/playlist/37i9dQZF1DWYxwmBaMqxsl"
}

# Playlist Display Names
PLAYLIST_NAMES = {
    "happy": "Happy Vibes 🎉",
    "sad": "Chill & Relax 🌙", 
    "neutral": "Focus Flow 🎧",
    "angry": "Energy Boost ⚡"
}