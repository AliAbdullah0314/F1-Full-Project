import React from 'react';
// import './RaceResults.css';
import {formatTime} from '../utils/timeUtils.js'

const RaceResults = ({ drivers, onClose }) => {
  // Sort drivers by final position
  const sortedDrivers = [...drivers].sort((a, b) => {
    const aProgress = a.current_lap + (a.lastProgress || 0);
    const bProgress = b.current_lap + (b.lastProgress || 0);
    return bProgress - aProgress;
  });

  const winner = sortedDrivers[0];

  return (
    <div className="race-results-overlay">
      <div className="race-results-modal">
        <h2>🏁 RACE FINISHED 🏁</h2>
        <h3>{winner.name} wins the race!</h3>
        
        <div className="podium">
          {sortedDrivers.slice(0, 3).map((driver, index) => (
            <div 
              key={index} 
              className={`podium-position podium-${index + 1}`}
              style={{ backgroundColor: driver.color + '22', borderColor: driver.color }}
            >
              <div className="position-number">{index + 1}</div>
              <div className="driver-name">{driver.name}</div>
            </div>
          ))}
        </div>

        <table className="final-standings">
          <thead>
            <tr>
              <th>Pos</th>
              <th>Driver</th>
              <th>Laps</th>
              
            </tr>
          </thead>
          <tbody>
            {sortedDrivers.map((driver, index) => (
              <tr key={index}>
                <td>{index + 1}</td>
                <td>
                  <span 
                    className="driver-color" 
                    style={{ backgroundColor: driver.color }}
                  ></span>
                  {driver.name}
                </td>
                <td>{driver.current_lap + 1}</td>
                
              </tr>
            ))}
          </tbody>
        </table>
        
      </div>
    </div>
  );
};

export default RaceResults;
