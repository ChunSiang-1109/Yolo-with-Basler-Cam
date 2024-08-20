import React from 'react';
import { useState, useEffect, useRef } from 'react';
import { styled } from '@mui/material/styles';
import { io, Socket } from 'socket.io-client';
import axios, { AxiosError } from 'axios';

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

  //output table for streaming stream
  const [detectionResult, setDetectionResult] = useState<any>(null);
  const [percentage, setPercentage] = useState<any>(null);

  //output table for non streaming mode
  const [detectionResultsList, setDetectionResultsList] = useState<any[]>([]);

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

  useEffect(() => {
    if (capturedImage) {
      console.log("Capture Image!!!");
      fetchImage();
      setCapturedImage(false);
    }
  }, [capturedImage]);
  
  useEffect(() => {
    console.log("Updated percentage:", percentage);
    setPercentage(percentage);
  }, [percentage]);
  
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
        // console.log("!!!!",data);
        videoRef.current.src = `data:image/jpeg;base64,${data.frame}`;
      }
      if(!buttonHide && videoRef.current){ 
        //streaming mode for detection result table
        setDetectionResult(data.result);
        startRecording();
        setPercentage(data.percentage || {}); // Ensure percentage is included
      }
      if(buttonHide && videoRef.current)
      {
        stopRecording();
      }
    });

    setSocketInstance(newSocket);
  };

  const startRecording = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/recordstream', {action: 'start'});
      console.log('Recording started:', response.data);
    } catch (error) {
        console.error('Error starting recording:', error);
        alert(`Error starting recording: ${error}`);
    }
  };
  
  const stopRecording = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/recordstream', { action: 'stop' });
      console.log('Recording stopped:', response.data);
    } catch (error) {
        console.error('Error stopping recording:', error);
    }
  };
  

  const fetchImage = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/get_last_capture_image');
      const { latest_image, detection_result ,percentage} = response.data;

      // Assuming latest_image is a base64 encoded string of the image
      if (latestCaptureRef.current) {
        latestCaptureRef.current.src = `data:image1/jpeg;base64,${latest_image}`;
      }
      console.log("Detection Result:", detection_result);
      console.log("Class Percentages:", percentage);
      
      setDetectionResult(detection_result);
      setPercentage(percentage);
      if(capturedImage && buttonHide){ 
        //non streaming mode for the cumulative detection result table
        setDetectionResultsList(prevResults => [...prevResults, detection_result]);
      }
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
    setDetectionResultsList([]);
  };

  const setReport = () => {
    fetch('http://127.0.0.1:5000/save_report', {
      method: 'GET',
    })
    .then(response => response.json())
    .then(data => {
      if (data.message) {
        console.log(data.message);  // Successfully saved
      } else {
        console.error(data.error);  // Error message
      }
    })
    .catch(error => {
      console.error('Error triggering report save:', error);
    });
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
        <DetectionResultTable detectionResult1={detectionResult} percentage1={percentage} />
        </Grid>
      </Grid>
      <BasicButtonGroup 
        setButtonStatus1={setStartStatus}
        StreamingEnabled={buttonHide}
        setButtonHide1={setButtonHide}
        setButtonCapOne={setCapturedImage}
        setReset={setReset}
        setSave={setReport}
         />
    </>
  );
}

export default App;
