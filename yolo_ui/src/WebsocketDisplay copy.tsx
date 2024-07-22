import React from 'react';
import { useState, useEffect, useRef } from 'react';

import { io, Socket } from 'socket.io-client';

// Now focus on this file

export default function WebsocketDisplay() {
  const [socket, setSocketInstance] = useState<Socket | null>(null);

  const [buttonStatus, setButtonStatus] = useState(false);
  const videoRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    return () => {
      if (socket) {
        socket.disconnect();
      }
    };
  }, [socket]);

  const connectSocket = () => {
    const newSocket = io('http://localhost:5000');

    newSocket.on('connect', () => {
      console.log("Frontend SocketIO connected!");

    });

    newSocket.on('disconnect', () => {
      console.log("Frontend SocketIO connected!");

    });

    newSocket.on('video_frame', (data) => {
      if (videoRef.current) {
        videoRef.current.src = `data:image/jpeg;base64,${data.frame}`;
      }
    });

    setSocketInstance(newSocket);
  };

  return (
    <React.Fragment>
      <button onClick={connectSocket}>Start Button Connect</button>
      <img ref={videoRef} alt="Webcam stream" />
    </React.Fragment>);
}