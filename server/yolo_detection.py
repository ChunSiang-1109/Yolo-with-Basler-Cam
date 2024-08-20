from ultralytics import YOLO
from flask import Flask,jsonify,request
from flask_cors import CORS
from flask_socketio import SocketIO
from pypylon import pylon
from ultralytics import YOLO
from pathlib import Path

from threading import Lock
import threading

import os
import cv2
import math
import time
import numpy as np
import openvino as ov
import torch
import base64
import datetime
import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

#to control the detection of the camera
detection_running= False
last_capture_frame = None
last_detection_result = None
last_class_percentages = None
recording=False
video_writer  = None
video_file_path = None
frame_width, frame_height = 640, 480
captured_images=[]
mutex_lock = Lock()

#initialize camera for stopgrabbing to restart the program
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

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

def calculate_class_percentages(detection_result):
    total_detections = sum(detection_result.values())
    if total_detections == 0:
        return {class_name: '0.00' for class_name in detection_result.keys()}

    class_percentages = {}
    for class_name, count in detection_result.items():
        percentage = (count / total_detections) * 100
        class_percentages[class_name] = f"{percentage:.2f}%"

    return class_percentages

def generate_and_save_report():
    global last_detection_result, last_class_percentages
    
    if not last_detection_result or not last_class_percentages:
        return {"error": "No data available to generate the report."}

    report_data = {
        "detectionResult": last_detection_result,
        "percentage": last_class_percentages
    }
    
    # Generate the timestamp for the filename
    malaysia_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    timestamp = malaysia_time.strftime("%Y%m%d_%H%M")

    # Set the filepath to Desktop/Kahang
    report_path = os.path.join(os.path.expanduser("~"), "Desktop", "Kahang","Report")
    # Ensure the directory exists
    os.makedirs(report_path, exist_ok=True)

    # Set the filename
    filename = f"Report_{timestamp}.json"
    filepath = Path(report_path) / filename

    # Save report data as JSON
    with open(filepath, 'w') as f:
        json.dump(report_data, f, indent=2)

    return {"message": f"Report saved successfully at {filepath}"}
    
def save_all_captured_images():
    if not captured_images:
        return {"error": "No images to save."}

    # Generate the timestamp for the folder name
    malaysia_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    timestamp = malaysia_time.strftime("%Y%m%d_%H%M")
    
    # Create a directory for saving images
    images_path = os.path.join(os.path.expanduser("~"), "Desktop", "Kahang", "Images", timestamp)
    os.makedirs(images_path, exist_ok=True)

    # Save all captured images
    for i, img in enumerate(captured_images):
        # Encode image as JPEG
        _, buffer = cv2.imencode('.jpg', img)
        filename = f"image_{i+1}.jpg"
        filepath = os.path.join(images_path, filename)
        with open(filepath, 'wb') as f:
            f.write(buffer)

    # Clear the list after saving
    captured_images.clear()
    
    return {"message": f"All images saved successfully in {images_path}"}

def save_video(frame_width, frame_height):
    # Generate the timestamp for the filename
    malaysia_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    timestamp = malaysia_time.strftime("%Y%m%d_%H%M")

    # Set the filepath to save the video
    video_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Kahang", "Video")
    os.makedirs(video_dir, exist_ok=True)
    video_file_path = os.path.join(video_dir, f"video_{timestamp}.mp4")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video_writer = cv2.VideoWriter(video_file_path, fourcc, 20.0, (frame_width, frame_height))
    
    return video_writer, video_file_path

def start_recording(frame_width, frame_height):
    global recording, video_writer, video_file_path
    
    if not recording:
        print("Recording is already in progress.")
        return
    
    video_writer, video_file_path = save_video(frame_width, frame_height)
    recording = True
    print(f"Recording started: {video_file_path}")

def stop_recording():
    global recording, video_writer,video_file_path

    if recording:
        print("No recording in progress.")
        return

    if video_writer is not None:
        video_writer.release()
        video_writer = None
        recording = False
        print(f"Recording stopped: {video_file_path}")

def video_stream():
    print("yolo_stream_detection")
    global start_time, frame_count, detection_running,stop_event, mutex_lock,video_writer
    global last_capture_frame, last_detection_result,last_class_percentages

    # Initialize the Basler camera interface
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        print("Error: No Basler camera found.")
        return

    models_dir = Path("./models")
    DET_MODEL_NAME = "best"

    # Specify the device as 'cuda' if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

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
                continue

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

                # Calculate class percentages
                eachclass_percentages  = calculate_class_percentages(detection_result)

                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', plot_image)
                frame_data = base64.b64encode(buffer).decode('utf-8')

                # Send frame over WebSocket
                socketio.emit('video_frame',{'frame':frame_data,'result':detection_result,'percentage':eachclass_percentages})
                last_capture_frame = frame_data
                last_detection_result = detection_result
                last_class_percentages = eachclass_percentages
                
                # Write frame to video file if recording
                if video_writer is not None:
                    video_writer.write(plot_image)

    finally:
        stop_event.set()
        camera1.StopGrabbing()
        camera1.Close()
        thread1.join()
        stop_recording()

#app instance
app=Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.post("/get_last_capture_image")
def send_last_capture_image():
    global mutex_lock, last_capture_frame, last_detection_result,last_class_percentages
    with mutex_lock:
        try:
            if last_capture_frame is not None:
                # Decode base64 to image
                img_data = base64.b64decode(last_capture_frame)
                np_arr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                captured_images.append(img)                

            response = {
                "latest_image": last_capture_frame,
                "detection_result": last_detection_result,
                "percentage": last_class_percentages
            }
            return jsonify(response), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/recordstream', methods=['POST'])
def record_stream():
    global recording, video_writer, video_file_path
    
    action = request.json.get('action')


    if action == 'start':
        if not recording:
            start_recording(frame_width, frame_height)
            return jsonify({"message": "Recording started."}), 200
        else:
            return jsonify({"error": "Recording already in progress."}), 400

    elif action == 'stop':
        if recording:
            stop_recording()
            return jsonify({"message": "Recording stopped."}), 200
        else:
            return jsonify({"error": "No recording in progress."}), 400

    else:
        return jsonify({"error": "Invalid action."}), 400

@app.route("/save_report", methods=["GET"])
def save_report():
    try:
        save_video(frame_width, frame_height)
        report_result = generate_and_save_report()
        image_result = save_all_captured_images()
        return jsonify({"report": report_result, "images": image_result}), 200 if "message" in report_result and "message" in image_result  else 400

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
    camera.StopGrabbing()
    mutex_lock.release()
    print("Client disconnected")

if __name__ == "__main__":

    try:
        threading.Thread(target=video_stream, daemon=True).start()
        socketio.run(app, host='127.0.0.1', port=5000)    
    except KeyboardInterrupt:
        pass

    cap.release()
