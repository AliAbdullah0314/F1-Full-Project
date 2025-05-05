import React from 'react';

const StandingsTable = ({ drivers }) => {
  if (!drivers || drivers.length === 0) {
    return (
      <div className="race-info">
        <h3>Live Race Standings</h3>
        <p>No driver data available</p>
      </div>
    );
  }

  // Calculate standings
  const calculateStandings = () => {
    const standings = drivers.map((driver, index) => {
      // Calculate progress exactly as in original code
      let totalLapTime = 0;
      for (let j = 0; j < driver.current_lap; j++) {
        totalLapTime += driver.lap_times[j] * 1000;
      }
      
      const timeInCurrentLap = driver.current_time - totalLapTime;
      const currentLapDuration = driver.lap_times[driver.current_lap] * 1000;
      const progressInLap = timeInCurrentLap / currentLapDuration;
      
      return {
        index,
        driver: driver.name,
        team: `${driver.current_lap + 1}`,
        color: driver.color,
        totalProgress: driver.current_lap + progressInLap,
        totalTime: driver.current_time
      };
    });
    
    // Sort by progress
    standings.sort((a, b) => b.totalProgress - a.totalProgress);
    
    // Calculate gaps
    return standings.map((driver, i) => {
      if (i === 0) {
        driver.position = 1;
        driver.gap = "Leader";
      } else {
        driver.position = i + 1;
        const progressDiff = standings[0].totalProgress - driver.totalProgress;
        const averageLapTime = drivers[driver.index].lap_times[drivers[driver.index].current_lap];
        const timeGap = progressDiff * averageLapTime;
        driver.gap = `+${timeGap.toFixed(1)}s`;
      }
      return driver;
    });
  };

  const standings = calculateStandings();

  return (
    <div className="race-info">
      <h3>Live Race Standings</h3>
      <table className="standings-table">
        <thead>
          <tr>
            <th>Pos</th>
            <th>Driver</th>
            <th>Lap</th>
            <th>Gap</th>
          </tr>
        </thead>
        <tbody>
          {standings.map((driver) => (
            <tr key={driver.index}>
              <td>{driver.position}</td>
              <td>
                <span 
                  className="team-color" 
                  style={{ backgroundColor: driver.color }}
                />
                {driver.driver}
              </td>
              <td>{driver.team}</td>
              <td>{driver.gap}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default StandingsTable;
