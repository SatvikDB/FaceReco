# 🎵 Spotify API Setup Guide for FaceReco

## 📋 Step-by-Step Spotify Developer Setup

### Step 1: Create Spotify Developer Account
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Log in with your Spotify account
3. Accept the Terms of Service

### Step 2: Create New App
1. Click **"Create an App"**
2. Fill in the details:
   - **App name**: `FaceReco Emotion Music`
   - **App description**: `AI-powered emotion detection with automatic music recommendation`
   - **Website**: Leave blank or use `https://github.com/SatvikDB/FaceReco`
   - **Redirect URI**: `https://localhost:8888/callback`

### Step 3: Configure App Settings
1. After creating the app, click on it to open settings
2. Click **"Edit Settings"**
3. In **Redirect URIs**, add:
   ```
   https://localhost:8888/callback
   ```
4. Click **"Add"** then **"Save"**

### Step 4: Get Your Credentials
1. In your app dashboard, you'll see:
   - **Client ID**: Copy this value
   - **Client Secret**: Click "Show Client Secret" and copy this value

## 🔐 **Recommended Redirect URIs:**

### **Primary (Use This One):**
```
https://localhost:8888/callback
```

### **Alternative Options:**
```
https://localhost:3000/callback
https://127.0.0.1:8888/callback
https://example.com/callback
```

## ⚙️ **Configure FaceReco with Your Credentials:**

### Method 1: Interactive Setup (Recommended)
```bash
python3 spotify_config.py
```
Enter your credentials when prompted.

### Method 2: Environment Variables
```bash
export SPOTIFY_CLIENT_ID="your_client_id_here"
export SPOTIFY_CLIENT_SECRET="your_client_secret_here"
export SPOTIFY_REDIRECT_URI="https://localhost:8888/callback"
```

### Method 3: Direct Configuration
Create a file called `spotify_credentials.py`:
```python
SPOTIFY_CLIENT_ID = "your_client_id_here"
SPOTIFY_CLIENT_SECRET = "your_client_secret_here"
SPOTIFY_REDIRECT_URI = "https://localhost:8888/callback"
```

## 🧪 **Test Your Setup:**

### Quick Test
```bash
python3 -c "
from spotify_integration import SpotifyController
controller = SpotifyController('YOUR_CLIENT_ID', 'YOUR_CLIENT_SECRET')
print('✅ Setup successful!' if controller.is_authenticated else '❌ Setup failed')
"
```

### Full System Test
```bash
python3 test_emotion_music_system.py
```

## 🎵 **Required Spotify Scopes:**
Our app requests these permissions:
- `user-read-playback-state` - Check what's currently playing
- `user-modify-playback-state` - Control playback (play/pause/skip)
- `playlist-read-private` - Access your playlists
- `user-read-currently-playing` - Get current track info

## 🔧 **Troubleshooting:**

### **"Invalid Redirect URI" Error**
- Make sure you added `https://localhost:8888/callback` exactly
- Check for typos in the URI
- Ensure you clicked "Save" after adding the URI

### **"Invalid Client" Error**
- Double-check your Client ID and Client Secret
- Make sure there are no extra spaces when copying
- Verify the credentials are from the correct app

### **"No Active Device" Error**
- Open Spotify on any device (phone, computer, web player)
- Start playing any song
- Run the FaceReco demo again

### **Authentication Timeout**
- Check your internet connection
- Try using a different redirect URI from the alternatives above
- Clear the Spotify cache: `rm .spotify_cache`

## 🎯 **What Happens During Authentication:**

1. **First Run**: Browser opens to Spotify login
2. **Login**: You log in with your Spotify account
3. **Authorize**: You grant permissions to FaceReco
4. **Redirect**: Spotify redirects to `https://localhost:8888/callback`
5. **Token Exchange**: FaceReco receives and stores access tokens
6. **Ready**: System can now control your Spotify playback

## 🔒 **Security Notes:**

- **Client Secret**: Keep this private, don't share it
- **Tokens**: Stored locally in `.spotify_cache` file
- **Permissions**: You can revoke access anytime in Spotify settings
- **Development Only**: This setup is for development/personal use

## 🎪 **Default Playlist Mapping:**

Once configured, FaceReco will use these playlists:

| Emotion | Default Playlist | Spotify URI |
|---------|------------------|-------------|
| 😊 Happy | Happy Hits! | `37i9dQZF1DXdPec7aLTmlC` |
| 😢 Sad | Sad Songs | `37i9dQZF1DX7qK8ma5wgG1` |
| 😐 Neutral | Chill Hits | `37i9dQZF1DWXRqgorJj26U` |
| 😠 Angry | Beast Mode | `37i9dQZF1DWYxwmBaMqxsl` |

You can customize these playlists using:
```bash
python3 spotify_config.py
```

## 🚀 **Ready to Use:**

After setup, run the full emotion-music demo:
```bash
python3 emotion_music_demo.py
```

Your FaceReco system will:
1. 🎥 Detect your face via camera
2. 🎭 Analyze your emotion
3. 🎵 Automatically play matching Spotify playlist
4. 📊 Display everything in real-time

---

**🎵 Need help? The setup should take less than 5 minutes! 🎵**