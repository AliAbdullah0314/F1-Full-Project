import React from 'react';
// import './PlaybackControls.css';

const PlaybackControls = ({ 
  isPlaying, 
  onTogglePlay, 
  progress, 
  onSeek, 
  onReset 
}) => {
  const handleProgressChange = (e) => {
    const newProgress = parseFloat(e.target.value);
    // console.log('newProgress:'+newProgress)
    onSeek(newProgress);
  };

  return (
    <div className="playback-controls">
      <div className="button-container">
        <button 
          className="play-button" 
          onClick={onTogglePlay}
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? '⏸️ Pause' : '▶️ Play'}
        </button>
        <button 
          className="reset-button" 
          onClick={onReset}
          aria-label="Reset"
        >
          🔄 Reset
        </button>
      </div>
      
      <div className="progress-container">
        <input
          type="range"
          min="0"
          max="100"
          value={progress || 0}
          onChange={handleProgressChange}
          className="progress-bar"
          aria-label="Race progress"
        />
        {/* <div className="progress-labels">
          
          <span>{Math.round(progress)}%</span>
          
        </div> */}
      </div>
    </div>
  );
};

export default PlaybackControls;
