import asyncio
import base64
import json
import websockets
from PIL import Image
import io

async def test_websocket():
    uri = "ws://localhost:8000/api/v1/stream"
    print(f"Connecting to {uri}...")
    
    # Create a dummy image to send
    img = Image.new('RGB', (224, 224), color = 'red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    async with websockets.connect(uri) as websocket:
        print("Connected!")
        # Send frame
        payload = f"data:image/jpeg;base64,{img_b64}"
        await websocket.send(payload)
        print("Sent frame, waiting for response...")
        
        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        
        print("\nSUCCESS! Received Response:")
        print(f"Faces Detected: {data.get('num_faces')}")
        print(f"Agentic Caption: {data.get('agentic_caption')}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
