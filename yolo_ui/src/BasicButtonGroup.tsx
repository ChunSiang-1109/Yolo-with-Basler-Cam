import React from 'react';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';


export default function BasicButtonGroup(
  { setButtonStatus1, setButtonCapOne, setReset, StreamingEnabled, setButtonHide1 }:
    { setButtonStatus1: Function, setButtonCapOne: Function, setReset: Function, StreamingEnabled: boolean, setButtonHide1: Function }) {

  function handleStart() {
    setButtonStatus1(true);
  }

  function handleStream() {
    if (StreamingEnabled) {
      setButtonHide1(false);
    }
    else {
      setButtonHide1(true); // Show the video
    }
  };

  function handleCaptureOneImage() {
    setButtonCapOne(true);
  }

  function handleReset() {
    setReset(true);
    setButtonHide1(false);
    setButtonStatus1(false);
  }

  return (
    <ButtonGroup variant="contained" aria-label="Basic button group">
      <Button onClick={handleStart} >Start</Button>
      <Button onClick={handleStream} >Stream</Button>
      <Button onClick={handleCaptureOneImage} >Capture One Image</Button>
      <Button onClick={handleReset} >Reset</Button>
    </ButtonGroup>
  );
}
