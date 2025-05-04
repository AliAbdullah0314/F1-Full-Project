/**
 * Process raw trajectory data into normalized track coordinates
 * @param {Array} trajectory - Raw trajectory data points
 * @param {number} scaleFactor - Factor to scale down the coordinates
 * @returns {Array} Processed track coordinates
 */
export const processTrackCoordinates = (trajectory, scaleFactor = 4) => {
    if (!trajectory || !trajectory.length) {
      console.error('Invalid trajectory data');
      return [];
    }
  
    const coordinates = [];
    for (let i = 0; i < trajectory.length; i += 2) {
      if (i + 1 < trajectory.length) {
        coordinates.push([
          trajectory[i] / scaleFactor, 
          trajectory[i + 1] / scaleFactor
        ]);
      }
    }
    
    return coordinates;
  };
  
  /**
   * Calculate position on track from progress percentage
   * @param {number} percentage - Progress around the track (0-1)
   * @param {Array} trackCoordinates - Array of track coordinate points
   * @returns {Array} [x, y] position
   */
  export const getPositionFromPercentage = (percentage, trackCoordinates) => {
    if (!trackCoordinates || trackCoordinates.length === 0) {
      return [0, 0];
    }
    
    // Validate and clamp percentage
    percentage = Math.max(0, Math.min(1, percentage || 0));
    
    const totalPoints = trackCoordinates.length;
    const exactIndex = percentage * totalPoints;
    const lowerIndex = Math.floor(exactIndex) % totalPoints;
    const upperIndex = (lowerIndex + 1) % totalPoints;
    const remainder = exactIndex - Math.floor(exactIndex);
    
    const lowerPoint = trackCoordinates[lowerIndex];
    const upperPoint = trackCoordinates[upperIndex];
    
    if (!lowerPoint || !upperPoint) {
      return [0, 0];
    }
    
    // Interpolate between points
    const x = lowerPoint[0] + (upperPoint[0] - lowerPoint[0]) * remainder;
    const y = lowerPoint[1] + (upperPoint[1] - lowerPoint[1]) * remainder;
    
    return [x, y];
  };
  
  /**
   * Calculate start/finish line coordinates
   * @param {Array} trackCoordinates - Array of track coordinate points
   * @param {number} lineLength - Length of the start/finish line
   * @returns {Array} [x1, y1, x2, y2] line coordinates
   */
  export const calculateStartLine = (trackCoordinates, lineLength = 20) => {
    if (!trackCoordinates || trackCoordinates.length < 2) {
      return [0, 0, 0, 0];
    }
    
    const startPoint = trackCoordinates[0];
    const directionPoint = trackCoordinates[1];
    
    // Calculate angle perpendicular to track direction
    const angle = Math.atan2(
      directionPoint[1] - startPoint[1],
      directionPoint[0] - startPoint[0]
    ) + Math.PI / 2;
    
    // Calculate line endpoints
    const startX1 = startPoint[0] + Math.cos(angle) * lineLength;
    const startY1 = startPoint[1] + Math.sin(angle) * lineLength;
    const startX2 = startPoint[0] - Math.cos(angle) * lineLength;
    const startY2 = startPoint[1] - Math.sin(angle) * lineLength;
    
    return [startX1, startY1, startX2, startY2];
  };
  