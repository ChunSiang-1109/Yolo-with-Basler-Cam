import React from 'react';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';


export default function BasicButtonGroup(
  { buttonStatus1, setButtonStatus1, buttonCapOne, setButtonCapOne }:
    { buttonStatus1: boolean, setButtonStatus1: Function, buttonCapOne: String|null, setButtonCapOne: Function }){

  function handleStart() {
    //Send request to Flask server
    // fetch('http://localhost:5000/api/start')
    //   .then(response => response.text())
    //   .then(data => console.log(data))
    //   .catch(error => console.error('Error:', error));
    setButtonStatus1(true);
    setButtonCapOne(null);
  };

  function handleStop() {
    //Send request to Flask server
    // fetch('http://localhost:5000/api/stop')
    //   .then(response => response.text())
    //   .then(data => console.log(data))
    //   .catch(error => console.error('Error:', error));
    setButtonStatus1(false);
  }

  function handleCaptureOneImage() {
    // fetch('http://localhost:5000/api/start')
    //   .then(response => {
    //     if (response.ok) {
    //       return response.json(); // Parse the response data as JSON
    //     } else {
    //       throw new Error('API request failed');
    //     }
    //   })
    //   .then(data => {
    //     // Process the response data here
    //     console.log(data);
    //   })
    //   .catch(error => {
    //     // Handle any errors here
    // //     console.error(error);
    // //   });
    setButtonCapOne(true);
  }

  return (
    <ButtonGroup variant="contained" aria-label="Basic button group">
      <Button onClick={handleStart} disabled={buttonStatus1}>Start</Button>
      <Button onClick={handleStop} disabled={!buttonStatus1}>Stop</Button>
      <Button onClick={handleCaptureOneImage} disabled={!buttonStatus1}>Capture One Image</Button>
      {/* <Button onClick={handleKeepCapturing}>Keep Capturing</Button> */}
    </ButtonGroup>
  );
}



// function handleKeepCapturing() {
//   fetch('http://localhost:5000/api/stop')
//     .then(response => {
//       if (response.ok) {
//         return response.json(); // Parse the response data as JSON
//       } else {
//         throw new Error('API request failed');
//       }
//     })

//     .then(data => {
//       // Process the response data here
//       console.log(data);
//     })

//     .catch(error => {
//       // Handle any errors here
//       console.error(error);
//     });
// }

