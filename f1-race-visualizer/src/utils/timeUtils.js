/**
 * Convert lap time string (e.g., "1:43.200") to milliseconds
 * @param {string} lapTimeStr - Lap time in format "M:SS.mmm" or "SS.mmm"
 * @returns {number} Time in milliseconds
 */
export const lapTimeToMs = (lapTimeStr) => {
    if (!lapTimeStr) return 0;
    
    try {
      const parts = lapTimeStr.split(':');
      const seconds = parseFloat(parts[parts.length - 1]);
      const minutes = parts.length > 1 ? parseInt(parts[0], 10) : 0;
      return (minutes * 60 + seconds) * 1000;
    } catch (e) {
      console.error('Error parsing lap time:', e);
      return 0;
    }
  };
  
  /**
   * Format milliseconds as lap time string
   * @param {number} ms - Time in milliseconds
   * @param {boolean} includeMs - Whether to include milliseconds
   * @returns {string} Formatted time string
   */
  export const formatTime = (ms, includeMs = true) => {
    if (typeof ms !== 'number' || isNaN(ms)) return '--:--';
    
    const totalSeconds = ms / 1000;
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = Math.floor(totalSeconds % 60);
    const milliseconds = Math.floor((totalSeconds % 1) * 1000);
    
    const formattedMinutes = String(minutes).padStart(1, '0');
    const formattedSeconds = String(seconds).padStart(2, '0');
    
    if (includeMs) {
      const formattedMs = String(milliseconds).padStart(3, '0');
      return `${formattedMinutes}:${formattedSeconds}.${formattedMs}`;
    }
    
    return `${formattedMinutes}:${formattedSeconds}`;
  };
  
  /**
   * Calculate time gaps between drivers
   * @param {Object} leader - Leader driver object
   * @param {Object} driver - Current driver object
   * @returns {string} Formatted gap time
   */
  export const calculateGap = (leader, driver) => {
    if (!leader || !driver) return '';
    
    if (leader === driver) return 'Leader';
    
    // Calculate lap difference
    const lapDifference = leader.current_lap - driver.current_lap;
    
    // If more than one lap behind, show laps
    if (lapDifference > 1) {
      return `+${lapDifference} laps`;
    }
    
    // Calculate time gap
    const leaderProgress = leader.current_lap + (leader.lastProgress || 0);
    const driverProgress = driver.current_lap + (driver.lastProgress || 0);
    const progressDiff = leaderProgress - driverProgress;
    
    // Convert progress difference to time (using current lap time as reference)
    const averageLapTime = driver.lap_times[driver.current_lap];
    const timeGap = progressDiff * averageLapTime;
    
    return `+${timeGap.toFixed(3)}s`;
  };
  
  /**
   * Parse CSV lap times data
   * @param {string} csvData - Raw CSV text content
   * @returns {Object} Object containing driver lap times
   */
  export const parseLapTimesCSV = (csvData) => {
    if (!csvData) return {};
    
    const lines = csvData.trim().split('\n');
    if (lines.length < 2) return {};
    
    const headers = lines[0].split(',');
    const totalLaps = headers.length - 1; // First column is driver name
    
    const drivers = [];
    
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      const driverName = values[0];
      
      if (!driverName) continue;
      
      const lapTimes = values.slice(1)
        .map(v => parseFloat(v))
        .filter(time => !isNaN(time));
      
      if (lapTimes.length > 0) {
        drivers.push({
          name: driverName,
          lap_times: lapTimes
        });
      }
    }
    
    return {
      drivers,
      totalLaps
    };
  };
  