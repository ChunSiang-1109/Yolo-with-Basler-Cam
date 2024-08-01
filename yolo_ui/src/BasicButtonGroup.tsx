import React from 'react';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import { Socket } from 'socket.io-client';


export default function BasicButtonGroup(
  { buttonStatus1, setButtonStatus1, setButtonRunStatus1  ,buttonCapOne, setButtonCapOne, socket ,setReset,setButtonHide1}:
    { buttonStatus1: boolean, setButtonStatus1: Function,setButtonRunStatus1:Function,  buttonCapOne: String|null, setButtonCapOne: Function, socket: Socket | null ,setReset:Function,setButtonHide1:Function}){
  
  function handleStart(){
    if (socket) {
      socket.emit('stop_stream');
      // Emit event to stop the stream
    }
    setButtonHide1(false); // Hide the video
    setButtonRunStatus1(true); 
    setButtonCapOne(null);
  }
  
  function handleStream() {
    // setButtonStatus1(true);
    // setButtonCapOne(null);
    if (socket) {
      socket.emit('start_detection'); // Emit event to start detection
    }
    // setButtonRunStatus1(true);
    setButtonStatus1(true); // Enable start button
    setButtonHide1(true); // Show the video
  };

  function handleCaptureOneImage() {
    // setButtonCapOne(true);
    if (socket) {
      socket.emit('start_capture');
      setButtonCapOne(true); 
    }
  }

  function handleReset() {
    if (socket) {
      socket.emit('clear_display'); // Emit event to capture image
      setReset(true); // Optionally reset captured image state here
    }
  }

  return (
    <ButtonGroup variant="contained" aria-label="Basic button group">
      <Button onClick={handleStart} >Start</Button>
      <Button onClick={handleStream} >Stream</Button>
      {/* <Button onClick={handleStop} disabled={!buttonStatus1}>Pause</Button> */}
      <Button onClick={handleCaptureOneImage} >Capture One Image</Button>
      <Button onClick={handleReset} >Reset</Button>
    </ButtonGroup>
  );
}
