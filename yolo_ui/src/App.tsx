import React from 'react';
import { useState, useEffect, useRef } from 'react';
import { styled } from '@mui/material/styles';
import { io, Socket } from 'socket.io-client';

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
  const [buttonStatus, setButtonStatus] = useState(false);
  //output video stream
  const videoRef = useRef<HTMLImageElement>(null);
  //output capturing one
  const latestCaptureRef = useRef<HTMLCanvasElement>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
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
    if (buttonStatus) {
      connectSocket();
    }
    else {
      disconnectSocket();
    }
  }, [buttonStatus]);

  useEffect(() => {
    if (capturedImage) {
      capImageOne();
 
    }
    else {
      disconnectSocket();
    }
  }, [capturedImage]);
  
  useEffect(() => {
    if (videoRef.current) {
      if (buttonHide) {
        videoRef.current.style.display = 'none';
      } else {
        videoRef.current.style.display = 'block';      
      }
    }
  }, [buttonHide]);

  const connectSocket = () => {
    const newSocket = io('http://localhost:5000');

    newSocket.on('connect', () => {
      console.log("Frontend SocketIO connected!");

    });

    newSocket.on('disconnect', () => {
      console.log("Frontend SocketIO disconnected!");

    });

    newSocket.on('video_frame', (data,detect_result) => {
      if (videoRef.current && !buttonHide) {
        videoRef.current.src = `data:image/jpeg;base64,${data.frame}`;
        setDetectionResult(data.result);
      }else{
        if(latestCaptureRef.current){
          setDetectionResult(data.result);
        }
      }
    });

    setSocketInstance(newSocket);
  };

  const disconnectSocket = () => {
    if (socket) {
      socket.disconnect();
      setSocketInstance(null);
    }
  };

  const capImageOne = () => {
    if (videoRef.current && latestCaptureRef.current) {
      const image = videoRef.current;
      const latestCapture = latestCaptureRef.current;
      const context = latestCapture.getContext("2d");

      if (context) {
        latestCapture.width = image.naturalWidth;
        latestCapture.height = image.naturalHeight;

        //draw video frame onto canvas
        context.drawImage(image, 0, 0, latestCapture.width, latestCapture.height);

        //get image data url from canvas
        const capturedImage1 = latestCapture.toDataURL("image/jpeg");

        //set the captured image
        setCapturedImage(capturedImage1);
      }
    }
  };

  const set_Reset = () => {
    if (videoRef.current) {
      videoRef.current.src = '';
    }
    if (latestCaptureRef.current) {
      const context = latestCaptureRef.current.getContext('2d');
      if (context) {
        context.clearRect(0, 0, latestCaptureRef.current.width, latestCaptureRef.current.height);
      }
    }
    setDetectionResult(null);
    setCapturedImage(null); // Clear captured image if needed
  };

  return (
    <>
      <Grid container spacing={5} justifyContent="center" alignItems="center">
        <Grid item xs={4}>
          <WebsocketDisplay image={videoRef} />
        </Grid>
        <Grid item xs={4}>
          <ShowCaptureImageDisplay image={latestCaptureRef} />
        </Grid>
        <Grid item xs={4}>
          <DetectionResultTable detectionResult={detectionResult} />
        </Grid>
      </Grid>
      <BasicButtonGroup
        buttonStatus1={buttonStatus}
        setButtonStatus1={setButtonStatus}
        buttonCapOne={capturedImage}
        setButtonCapOne={setCapturedImage}
        socket={socket}
        setReset={set_Reset}
        setButtonHide1={setButtonHide} />
    </>
  );
}

export default App;
