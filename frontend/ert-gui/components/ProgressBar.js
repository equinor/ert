import React from 'react';

const ProgressBar = ({ progress }) => {
  return (
    <div style={{ border: '1px solid #ccc', width: '100%' }}>
      <div
        style={{
          height: '24px',
          width: `${progress}%`,
          backgroundColor: 'green',
          textAlign: 'center',
          color: 'white'
        }}
      >
        {progress}%
      </div>
    </div>
  );
};

export default ProgressBar;
