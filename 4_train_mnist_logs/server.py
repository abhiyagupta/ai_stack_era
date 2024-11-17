from http.server import SimpleHTTPRequestHandler, HTTPServer
import os
import json
import asyncio
#import websockets

# Attempt to install the websockets library if not already installed
try:
    import websockets
except ImportError:
    print("websockets library not found. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'websockets'])
    import websockets  # Import again after installation

from http.server import SimpleHTTPRequestHandler,

HOST = "localhost"
PORT = 8080

class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Change the directory to 'static' for serving files
        super().__init__(*args, directory="static", **kwargs)
    
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

async def send_metrics(websocket, path):
    while True:
        try:
            with open('static/logs.json', 'r') as f:
                metrics = json.load(f)
            # Send the metrics to the client
            await websocket.send(json.dumps(metrics))
            print("[INFO] Sent metrics:", metrics)  # Log sent metrics
        except Exception as e:
            print(f"[ERROR] Failed to read logs.json: {e}")
        
        await asyncio.sleep(1)  # Adjust the frequency of updates as needed

async def main():
    # Start the WebSocket server
    websocket_server = websockets.serve(send_metrics, HOST, PORT + 1)  # Use a different port for WebSocket
    await websocket_server

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    print(f"Starting HTTP server at http://{HOST}:{PORT}")
    print(f"Serving files from {os.path.abspath('static')}")
    
    # Start the HTTP server
    httpd = HTTPServer((HOST, PORT), CustomHandler)
    
    # Start the WebSocket server
    asyncio.run(main())
    
    # Serve HTTP requests
    httpd.serve_forever()