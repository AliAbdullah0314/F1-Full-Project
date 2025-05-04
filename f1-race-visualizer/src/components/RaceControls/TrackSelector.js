import React from 'react';
// import './TrackSelector.css';


const TrackSelector = ({ selectedTrack, onTrackChange }) => {
  const tracks = [
    { id: 'bahrain', name: 'Bahrain Grand Prix' },
    { id: 'austin', name: 'Austin Grand Prix' },
    { id: 'abu_dhabi', name: 'Abu Dhabi Grand Prix' }
  ];

  return (
    <div className="track-selector">
      <h3>Select Race Track</h3>
      <select 
        value={selectedTrack} 
        onChange={(e) => onTrackChange(e.target.value)}
        className="track-dropdown"
      >
        {tracks.map(track => (
          <option key={track.id} value={track.id}>
            {track.name}
          </option>
        ))}
      </select>
    </div>
  );
};

export default TrackSelector;
