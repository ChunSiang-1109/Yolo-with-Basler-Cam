import React from 'react';
import { useState, useEffect, useRef } from 'react';
import { io, Socket } from 'socket.io-client';

import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import { styled } from '@mui/material/styles';
import WebsocketDisplay from './WebsocketDisplay';
import BasicButtonGroup from './BasicButtonGroup';
import ShowCaptureImageDisplay from './ShowCaptureImageDisplay';

const Item = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  textAlign: 'center',
  width: '300px',
  height: '300px',
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
  //record the

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
        videoRef.current.src = `data:image/jpeg;base64,${data.frame}`;
      }
    });
    setSocketInstance(newSocket);
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

        //do the save captured image (save it to state or server)
      }
    }
  };



  const disconnectSocket = () => {
    if (socket) {
      socket.disconnect();
      setSocketInstance(null);
    }
  };


  return (
    <>
      <Grid container spacing={5} justifyContent="center" alignItems="center">
        <Grid item xs={6}>
          <WebsocketDisplay image={videoRef} />
        </Grid>
        <Grid item xs={6}>
          <ShowCaptureImageDisplay image={latestCaptureRef} />
        </Grid>
      </Grid>
      <BasicButtonGroup 
        buttonStatus1={buttonStatus} 
        setButtonStatus1={setButtonStatus}
        buttonCapOne={capturedImage} 
        setButtonCapOne={setCapturedImage}/>
    </>
  );
}

export default App;
