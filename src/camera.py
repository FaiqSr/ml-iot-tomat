import cv2
import socketio
import base64
import time

# Ganti dengan IP Laptop Server Flask
SERVER_URL = 'http://localhost:5000' 
DEVICE_ID = '1'

sio = socketio.Client()

@sio.event
def connect():
    print(f"✅ Terhubung ke Server sebagai {DEVICE_ID}")

@sio.event
def disconnect():
    print("❌ Terputus")

def main():
    sio.connect(SERVER_URL)
    
    # Buka Webcam (0)
    cap = cv2.VideoCapture(0) 
    # Atur resolusi rendah agar cepat (opsional)
    cap.set(3, 640) 
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Encode ke JPEG lalu ke Base64
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Kirim ke Server Flask
        sio.emit('send_image', {
            'device_id': DEVICE_ID,
            'image': img_str
        })

        # Limit FPS agar tidak membanjiri server (misal 15-20 FPS)
        time.sleep(0.05) 

if __name__ == '__main__':
    main()