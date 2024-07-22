import React from 'react';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import { styled } from '@mui/material/styles';
import WebsocketDisplay from './WebsocketDisplay';

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

// Function to create the basic grid
const createBasicGrid = () => {
  return (
    <Grid container spacing={5} justifyContent="center" alignItems="center">
      <Grid item xs={6}>
        {/* <WebsocketDisplay/> */}
      </Grid>
      <Grid item xs={6}>
      <Item>Right Grid</Item>
      </Grid>
    </Grid>
  );
};

const BasicGrid: React.FC = () => {
  return (
    <div>
      {createBasicGrid()}
    </div>
  );
};

export default BasicGrid;