import React from 'react';
import { useState, useEffect, useRef } from 'react';
import { styled } from '@mui/material/styles';
import { io, Socket } from 'socket.io-client';
import axios from 'axios';

import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';

import WebsocketDisplay from './WebsocketDisplay';
import BasicButtonGroup from './BasicButtonGroup';
import ShowCaptureImageDisplay from './ShowCaptureImageDisplay';
import DetectionResultTable from './DetectionResultTable';


const Item = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  textAlign: 'center',
  width: '200px',
  height: '200px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: theme.palette.text.secondary,
}));

function App() {
  //to start or stop 
  const [socket, setSocketInstance] = useState<Socket | null>(null);
  //button setting
  const [startStatus, setStartStatus] = useState(false);

  //output video stream
  const videoRef = useRef<HTMLImageElement>(null);
  //output capturing one
  const latestCaptureRef = useRef<HTMLImageElement>(null);

  const [capturedImage, setCapturedImage] = useState(false);

  //output table for video stream
  const [detectionResult, setDetectionResult] = useState<any>(null);
  //button run
  const [buttonHide, setButtonHide] = useState<boolean>(false);


  useEffect(() => {
    return () => {
      if (socket) {
        socket.disconnect();
      }
    };
  }, [socket]);

  useEffect(() => {
    if (startStatus) {
      connectSocket();
    }
    else {
      disconnectSocket();
    }
  }, [startStatus]);


  const connectSocket = () => {
    const newSocket = io('http://localhost:5000');

    newSocket.on('connect', () => {
      console.log("Frontend SocketIO connected!");
    });

    newSocket.on('disconnect', () => {
      console.log("Frontend SocketIO disconnected!");
    });

    newSocket.on('video_frame', (data) => {
      if (videoRef.current) {
        console.log("!!!!");
        videoRef.current.src = `data:image/jpeg;base64,${data.frame}`;
        // setDetectionResult(data.result);
      }
    });

    newSocket.on('detection_result', (detect_result) => {
        setDetectionResult(detect_result.result);
    });

    setSocketInstance(newSocket);
  };

  useEffect(() => {
    if (capturedImage) {
      console.log("Capture Image!!!");
      fetchImage();
      setCapturedImage(false);
    }
  }, [capturedImage]);

  const fetchImage = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/get_last_capture_image');
      const { latest_image, detection_result } = response.data;

      // Assuming latest_image is a base64 encoded string of the image
      if (latestCaptureRef.current) {
        latestCaptureRef.current.src = `data:image1/jpeg;base64,${latest_image}`;
      }
      console.log(detection_result);

      setDetectionResult(detection_result);
    } catch (error) {
      console.error("Error fetching the image:", error);
    }
  };

  const disconnectSocket = () => {
    if (socket) {
      socket.disconnect();
      setSocketInstance(null);
    }
  };

  const setReset = () => {
    if (videoRef.current) {
      videoRef.current.src = '';
    }
    if (latestCaptureRef.current) {
      latestCaptureRef.current.src = '';
      }
    setDetectionResult({});
  };

  return (
    <>
      <Grid container spacing={5} justifyContent="center" alignItems="center">
        <Grid item xs={4}>
          <WebsocketDisplay showStreamingImage={buttonHide} image={videoRef} />
        </Grid>
        <Grid item xs={4}>
          <ShowCaptureImageDisplay image1={latestCaptureRef} />
        </Grid>
        <Grid item xs={4}>
        <DetectionResultTable detectionResult={detectionResult} />
        </Grid>
      </Grid>
      <BasicButtonGroup
        setButtonStatus1={setStartStatus}
        setButtonCapOne={setCapturedImage}
        setReset={setReset}
        StreamingEnabled={buttonHide}
        setButtonHide1={setButtonHide} />
    </>
  );
}


export default App;
