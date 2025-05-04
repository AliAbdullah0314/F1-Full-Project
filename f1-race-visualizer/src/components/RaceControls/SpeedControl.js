import React from 'react';
// import './SpeedControl.css';

const SpeedControl = ({ speed, onSpeedChange }) => {
  const handleSpeedChange = (e) => {
    const newSpeed = parseFloat(e.target.value);
    onSpeedChange(newSpeed);
  };

  return (
    <div className="speed-control">
      <label htmlFor="speedSlider">
        Simulation Speed: <span className="speed-value">{speed}x</span>
      </label>
      <input
        type="range"
        id="speedSlider"
        min="0.5"
        max="10"
        step="0.5"
        value={speed}
        onChange={handleSpeedChange}
        className="speed-slider"
      />
      <div className="speed-marks">
        
      </div>
    </div>
  );
};

export default SpeedControl;
