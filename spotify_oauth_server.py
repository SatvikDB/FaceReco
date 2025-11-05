#!/usr/bin/env python3
"""
🎵 Spotify OAuth Server
Local server to handle Spotify OAuth callback
"""

import threading
import webbrowser
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyCallbackHandler(BaseHTTPRequestHandler):
    """Handle Spotify OAuth callback"""
    
    def do_GET(self):
        """Handle GET request from Spotify redirect"""
        try:
            # Parse the callback URL
            parsed_url = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            if 'code' in query_params:
                # Success - got authorization code
                auth_code = query_params['code'][0]
                self.server.auth_code = auth_code
                self.server.auth_success = True
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                success_html = """
                <html>
                <head><title>FaceReco - Spotify Connected!</title></head>
                <body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1>🎉 Success!</h1>
                    <h2>Spotify Connected to FaceReco</h2>
                    <p>You can now close this window and return to the FaceReco demo.</p>
                    <p>Your emotions will control your Spotify music! 🎭🎵</p>
                </body>
                </html>
                """
                self.wfile.write(success_html.encode())
                
                logger.info("✅ Spotify authorization successful!")
                
            elif 'error' in query_params:
                # Error in authorization
                error = query_params['error'][0]
                self.server.auth_error = error
                self.server.auth_success = False
                
                # Send error response
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                error_html = f"""
                <html>
                <head><title>FaceReco - Spotify Error</title></head>
                <body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1>❌ Authorization Error</h1>
                    <p>Error: {error}</p>
                    <p>Please try again or check your Spotify app settings.</p>
                </body>
                </html>
                """
                self.wfile.write(error_html.encode())
                
                logger.error(f"❌ Spotify authorization error: {error}")
            
        except Exception as e:
            logger.error(f"❌ Error handling callback: {e}")
            self.send_response(500)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default HTTP server logging"""
        pass

class SpotifyOAuthServer:
    """Local server for Spotify OAuth flow"""
    
    def __init__(self, port=8888):
        self.port = port
        self.server = None
        self.server_thread = None
        self.auth_code = None
        self.auth_success = False
        self.auth_error = None
    
    def start_server(self):
        """Start the OAuth callback server"""
        try:
            self.server = HTTPServer(('127.0.0.1', self.port), SpotifyCallbackHandler)
            self.server.auth_code = None
            self.server.auth_success = False
            self.server.auth_error = None
            
            # Start server in background thread
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            logger.info(f"🌐 OAuth server started on http://127.0.0.1:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start OAuth server: {e}")
            return False
    
    def stop_server(self):
        """Stop the OAuth callback server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("🛑 OAuth server stopped")
    
    def wait_for_callback(self, timeout=60):
        """Wait for OAuth callback"""
        import time
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.server and hasattr(self.server, 'auth_success'):
                if self.server.auth_success:
                    self.auth_code = self.server.auth_code
                    return True
                elif hasattr(self.server, 'auth_error') and self.server.auth_error:
                    self.auth_error = self.server.auth_error
                    return False
            
            time.sleep(0.5)
        
        logger.warning("⏰ OAuth callback timeout")
        return False

def test_oauth_server():
    """Test the OAuth server"""
    print("🧪 Testing Spotify OAuth Server")
    
    server = SpotifyOAuthServer()
    
    if server.start_server():
        print("✅ Server started successfully")
        print("🌐 Test URL: http://127.0.0.1:8888/callback?code=test123")
        print("Press Ctrl+C to stop")
        
        try:
            import time
            time.sleep(30)  # Keep server running for 30 seconds
        except KeyboardInterrupt:
            pass
        
        server.stop_server()
        print("✅ Server stopped")
    else:
        print("❌ Failed to start server")

if __name__ == "__main__":
    test_oauth_server()