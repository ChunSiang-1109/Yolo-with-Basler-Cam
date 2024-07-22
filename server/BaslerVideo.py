from ultralytics import YOLO
from flask import Flask
from flask_socketio import SocketIO
import cv2
import time
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
    DET_MODEL_NAME = "yolov8x"
    det_model = YOLO(models_dir / f"{DET_MODEL_NAME}.pt")
    res = det_model(IMAGE_PATH)
    det_model_path = models_dir / f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
    core = ov.Core()
    device=core.available_devices[1]
    print(device)
    det_ov_model = core.read_model(det_model_path)
    
    ov_config = {}
    if device != "CPU":
        det_ov_model.reshape({0: [1, 3, 544, 640]})
    # if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
    #     ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    compiled_model = core.compile_model(det_ov_model, device, ov_config)
    
    def infer(*args):
            result = compiled_model(args)
            return torch.from_numpy(result[0])
    
    det_model.predictor.inference = infer

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


#app instance
app=Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on("connect")
def connect(*args, **kwargs):
    # reject all connections to test
    return "Connected"

@socketio.on("disconnect")
def disconnect(*args, **kwargs):
    # reject all connections to test
    return "Disconnected"

if __name__ == "__main__":
    try:
        # NOTE: Use Thread to solve the issue, there might be a better solution without multiple thread
        threading.Thread(target=video_stream, daemon=True).start()
        socketio.run(app, host='127.0.0.1', port=5000)    
    except KeyboardInterrupt:
        pass

    # Release resources
    cap.release()
    # cv2.destroyAllWindows()







