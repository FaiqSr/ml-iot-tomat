import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room
from ultralytics import YOLO

# Inisialisasi Flask & SocketIO
# async_mode='threading' PENTING agar tidak bentrok dengan PyTorch/YOLO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global State untuk Frame Dropping (Mencegah Lag)
is_processing = {} 

# --- 1. LOAD MODEL YOLO ---
# Pastikan file model ada. Jika tidak, akan download otomatis.
YOLO_MODEL_PATH = 'best.pt' 
print(f"⏳ Loading YOLO model ({YOLO_MODEL_PATH})...")
try:
    model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading YOLO: {e}")
    model = None # Handle gracefully

# --- HELPER FUNCTIONS ---

def base64_to_image(base64_string):
    """Decode Base64 string ke OpenCV Image"""
    try:
        # Buang header data:image/... jika ada
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def image_to_base64(img):
    """Encode OpenCV Image ke Base64 string (tanpa header)"""
    # Kompresi JPEG quality 50% agar ringan di network
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

# --- SOCKET EVENTS ---

@socketio.on('connect')
def handle_connect():
    print(f'🔌 Client Connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'❌ Client Disconnected: {request.sid}')

# 1. Viewer (Laravel) Join Room
@socketio.on('join_monitor')
def handle_join_monitor(data):
    device_id = data.get('device_id')
    if device_id:
        join_room(f"room_{device_id}")
        print(f"👀 Viewer {request.sid} watching {device_id}")
        # Kirim konfirmasi ke client bahwa dia sudah join
        emit('monitor_joined', {'device_id': device_id, 'status': 'ok'})

# 2. Alat (Camera) Kirim Video
@socketio.on('send_image')
def handle_video_stream(data):
    device_id = data.get('device_id', 'unknown')
    image_data = data.get('image') # Base64 string

    if not image_data or model is None: 
        return

    # FRAME DROPPING: Jika server masih sibuk proses frame sebelumnya, abaikan frame ini
    if is_processing.get(device_id, False):
        return
    
    is_processing[device_id] = True

    try:
        # Decode
        frame = base64_to_image(image_data)
        if frame is None: return

        # YOLO Inference (Conf 0.4 agar tidak terlalu sensitif)
        results = model(frame, verbose=False, conf=0.4, iou=0.5)
        
        # Annotate Frame
        annotated_frame = results[0].plot()

        # Encode Balik ke Base64
        output_base64 = image_to_base64(annotated_frame)

        # Broadcast ke Room Device ID
        # Kita kirim RAW base64 string TANPA header "data:image..."
        # Header akan ditambahkan di Client Side (JavaScript)
        emit('stream_frame', {
            'device_id': device_id,
            'image': output_base64
        }, room=f"room_{device_id}")

    except Exception as e:
        print(f"Error processing frame: {e}")
    
    finally:
        is_processing[device_id] = False

# --- RUN SERVER ---
if __name__ == '__main__':
    # allow_unsafe_werkzeug=True diperlukan di beberapa environment dev
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)