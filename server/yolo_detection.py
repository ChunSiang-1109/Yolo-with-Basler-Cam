from ultralytics import YOLO
from flask import Flask,jsonify
from flask_socketio import SocketIO
import cv2
import math
import time
import numpy as np
import threading
from pypylon import pylon
from ultralytics import YOLO
from pathlib import Path
import openvino as ov
import torch
import base64


# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Model
model = YOLO("yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# FPS calculation variables
start_time = time.time()
frame_count = 0


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
    # Initialize the Basler camera interface
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        print("Error: No Basler camera found.")
        return
    
    IMAGE_PATH = Path("./data/1.jpg")
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
    stop_event = threading.Event()

    thread1 = threading.Thread(target=capture_and_store, args=(camera1, converter, image_storage, 0, stop_event))

    thread1.start()

    frame_count = 0
    prev_time = time.time()

    try:
        while True:
            if image_storage[0] is not None:
                stitched_image = image_storage[0].copy()
                results = det_model.track(stitched_image, persist=True)

                for result in results:
                    plot_image = result.plot()
                    if plot_image is not None:
                        cv2.imshow('Object Detection', plot_image)

                frame_count += 1
                total_time = time.time() - prev_time

                if total_time >= 1.0:
                    fps = frame_count / total_time
                    print(f"FPS: {fps:.2f}")
                    frame_count = 0
                    prev_time = time.time()
                
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', plot_image)
                    frame_data = base64.b64encode(buffer).decode('utf-8')

                    # Send frame over WebSocket
                    socketio.emit('video_frame',{'frame':frame_data})

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stop_event.set()
        camera1.StopGrabbing()
        camera1.Close()
        cv2.destroyAllWindows()
        thread1.join()






        # # Calculate and display FPS
        # frame_count += 1
        # elapsed_time = time.time() - start_time
        # fps = frame_count / elapsed_time
        # cv2.putText(img, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # # Show webcam feed
        # cv2.imshow('Webcam', img)
        # # Quit if 'q' is pressed
        # if cv2.waitKey(1) == ord('q'):
        #     break


        # Control frame rate
        # await asyncio.sleep(0.04)  # Adjust to match video FPS


#app instance
app=Flask(__name__)
# socketio = SocketIO(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# @app.route("/api/start",methods=['GET'])
# def detect_start():
#     return jsonify({
#         'message': "Start"
#     })

# @app.route("/api/stop",methods=['GET'])
# def detect_stop():
#     return jsonify({
#         'message': "Stop"
#     })

@socketio.on("connect")
def connect(*args, **kwargs):
    # reject all connections to test
    return "Connected"

@socketio.on("disconnect")
def disconnect(*args, **kwargs):
    # reject all connections to test
    return "Disconnected"

if __name__ == "__main__":

    # app.run(debug=True)

    # start_server = websockets.serve(video_stream, "localhost", 1357)
    # asyncio.get_event_loop().run_until_complete(start_server)
    # asyncio.get_event_loop().run_forever()

    try:
        # NOTE: Use Thread to solve the issue, there might be a better solution without multiple thread
        threading.Thread(target=video_stream, daemon=True).start()
        socketio.run(app, host='127.0.0.1', port=5000)    
    except KeyboardInterrupt:
        pass

    # Release resources
    cap.release()
    # cv2.destroyAllWindows()

