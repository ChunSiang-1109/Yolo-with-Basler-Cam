from ultralytics import YOLO
from flask import Flask,jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import math
import time
import numpy as np

import threading
from threading import Lock

from pypylon import pylon
from ultralytics import YOLO
from pathlib import Path
import openvino as ov
import torch
import base64

#to control the detection of the camera
detection_running= False
last_capture_frame = None
last_detection_result = None
mutex_lock = Lock()

# Stop event for stopping the detection loop
stop_event = threading.Event()

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

def capture_image_from_basler_camera(camera, converter):
    grabResult = camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        img = image.GetArray()
        grabResult.Release()
        img = resize_with_aspect_ratio(img, width=833)
        return img
    else:
        grabResult.Release()
        return None

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

def capture_and_store(camera, converter, storage, index, stop_event):
    while not stop_event.is_set() and camera.IsGrabbing():
            img = capture_image_from_basler_camera(camera, converter)
            storage[0] = img
    
def video_stream():
    print("yolo_stream_detection")
    global start_time, frame_count, detection_running,stop_event, mutex_lock
    global last_capture_frame, last_detection_result

    # Initialize the Basler camera interface
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        print("Error: No Basler camera found.")
        return

    models_dir = Path("./models")
    DET_MODEL_NAME = "best"
    det_model = YOLO(models_dir / f"{DET_MODEL_NAME}.pt")

    camera1 = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
    camera1.Open()
    camera1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    image_storage = [None]
    stop_event.clear()

    thread1 = threading.Thread(target=capture_and_store, args=(camera1, converter, image_storage, 0, stop_event))
    thread1.start()

    frame_count = 0
    prev_time = time.time()

    # Table for results
    classes = det_model.names
    name = classes.values() 
    detection_result = {class_id: 0 for class_id in name}
    list = []

    try:
        while True:
            if not detection_running:
                # time.sleep(1)
                continue

        # try:
            if image_storage[0] is not None:
                stitched_image = image_storage[0].copy()
                results = det_model.track(stitched_image, persist=True)

                for result in results:
                    plot_image = result.plot()
                    
                    # Increment class counts
                    for box in result.boxes:
                        class_id = int(box.cls)
                        obj_id = box.id
                        pred_class = classes[class_id]
                        
                        if obj_id not in list:
                            list.append(obj_id)

                            if pred_class in detection_result.keys():
                                detection_result[pred_class] += 1
                                
                frame_count += 1
                total_time = time.time() - prev_time
                frame_count = 0
                prev_time = time.time()
            
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', plot_image)
                frame_data = base64.b64encode(buffer).decode('utf-8')

                # Send frame over WebSocket
                socketio.emit('video_frame',{'frame':frame_data,'result':detection_result})
                last_capture_frame = frame_data
                last_detection_result = detection_result
                
                if stop_event.is_set():
                    socketio.emit('clear_display')
                    break

    finally:
        stop_event.set()
        camera1.StopGrabbing()
        camera1.Close()
        thread1.join()

    
#app instance
app=Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.post("/get_last_capture_image")
def send_last_capture_image():
    global mutex_lock, last_capture_frame, last_detection_result
    with mutex_lock:
        try:
            response = {
                "latest_image": last_capture_frame,
                "detection_result": last_detection_result
            }
            return jsonify(response), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
@socketio.on('connect')
def handle_connect():
    global detection_running, mutex_lock
    mutex_lock.acquire()
    detection_running = True
    mutex_lock.release()
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    global detection_running, stop_event, mutex_lock
    mutex_lock.acquire()
    detection_running = False
    stop_event.set()
    socketio.emit('clear_display')
    mutex_lock.release()
    print("Client disconnected")

if __name__ == "__main__":

    try:
        threading.Thread(target=video_stream, daemon=True).start()
        socketio.run(app, host='127.0.0.1', port=5000)    
    except KeyboardInterrupt:
        pass

    cap.release()
