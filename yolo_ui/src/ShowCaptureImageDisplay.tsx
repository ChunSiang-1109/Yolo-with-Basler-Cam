import React from 'react';
import {  RefObject } from 'react';

export default function ShowCaptureImageDisplay({image}: {image: RefObject<HTMLCanvasElement>}) {
  return (
    <React.Fragment>
      <canvas ref={image}   style={{ width: '100%', height: 'auto' }}/>
    </React.Fragment>);
}