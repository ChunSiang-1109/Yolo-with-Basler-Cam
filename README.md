# Yolo-with-Basler-Cam

## Frontend

1. ``cd yolo_ui``
2. ``npm start``

## Backend

1. ``cd server``
2. ``python BaslerVideo.py`` /``python yolo_detection.py``

## Structure of program
1.Implement yolo_detection and Basler Camera by using Flask API as the backend server.

2.'capture_image_from_basler_camera' , ' resize_with_aspect_ratio ' and ' capture_and_store' functions have been implemented in these two python folders.

3.Run flask server using SocketIo with threading. SocketIO enables real-time communication between clients and servers.

4.For frontend UI, 'connectSocket' is defined to call the flask API and the console output will response "Frontend SocketIO connected!" when clicking the 'Start' 
  button. 

5.Once the 'Start' button clicked, the video streaming will display on UI

6.'Stop' button is to stop the video streaming and disconnect with the flask API.

7.'CapturueOneImage` is to capture the current video streaming image and then display at the right frame of UI.
