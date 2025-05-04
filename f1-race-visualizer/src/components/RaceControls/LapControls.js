import React from 'react';
// import './LapControls.css';

const LapControls = ({ totalLaps, onSeekToLap, currentLap }) => {
  // Create array of lap numbers
  const laps = Array.from({ length: totalLaps-1 }, (_, i) => i + 1);
  
  return (
    <div className="lap-controls">
      <h3>Race Laps</h3>
      <div className="lap-chapters">
        {laps.map((lap) => (
          <button
            key={lap}
            className={`lap-chapter ${currentLap === lap ? 'active' : ''}`}
            onClick={() => onSeekToLap(lap)}
          >
            Lap {lap}
          </button>
        ))}
      </div>
    </div>
  );
};

export default LapControls;
