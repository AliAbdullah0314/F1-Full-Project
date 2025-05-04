import React, { useRef, useEffect } from 'react';
import { Stage, Layer, Line, Circle, Text } from 'react-konva';
import RaceResults from './RaceResults';

const TrackVisualizer = ({ trackCoordinates, drivers, isPlaying, raceFinished }) => {
  const stageRef = useRef(null);
  
  // Convert track coordinates to Konva format
  const trackPoints = trackCoordinates.flatMap(point => [point[0], point[1]]);
  
  // Draw drivers on track
  const renderDrivers = () => {
    return drivers.map((driver, index) => {
      if (!driver.lastProgress) return null;
      
      const position = getPositionFromPercentage(driver.lastProgress, trackCoordinates);
      
      return (
        <React.Fragment key={index}>
          <Circle
            x={position[0]}
            y={position[1]}
            radius={12}
            fill={driver.color}
            stroke="white"
            strokeWidth={2}
          />
          <Text
            x={position[0] - 6}
            y={position[1] - 7}
            text={driver.name.charAt(0)}
            fill="black"
            fontSize={15}
            fontStyle = 'bold'
            align="center"
            stroke="white"
            strokeWidth = {0.3}
            
          />
        </React.Fragment>
      );
    });
  };
  
  // Resize handler for responsiveness
  useEffect(() => {
    const handleResize = () => {
      const container = document.querySelector('.track-container');
      if (container && stageRef.current) {
        const scale = Math.min(
          container.offsetWidth / 800,
          container.offsetHeight / 500
        );
        
        stageRef.current.width(800 * scale);
        stageRef.current.height(500 * scale);
        stageRef.current.scale({ x: scale, y: scale });
      }
    };
    
    window.addEventListener('resize', handleResize);
    handleResize();
    
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Calculate position from percentage
  const getPositionFromPercentage = (percentage, coords) => {
    if (!coords || coords.length === 0) return [0, 0];
    
    percentage = Math.max(0, Math.min(1, percentage || 0));
    
    const totalPoints = coords.length;
    const exactIndex = percentage * totalPoints;
    const lowerIndex = Math.floor(exactIndex) % totalPoints;
    const upperIndex = (lowerIndex + 1) % totalPoints;
    const remainder = exactIndex - Math.floor(exactIndex);
    
    const lowerPoint = coords[lowerIndex];
    const upperPoint = coords[upperIndex];
    
    if (!lowerPoint || !upperPoint) return [0, 0];
    
    const x = lowerPoint[0] + (upperPoint[0] - lowerPoint[0]) * remainder;
    const y = lowerPoint[1] + (upperPoint[1] - lowerPoint[1]) * remainder;
    
    return [x, y];
  };
  
  return (
    <div className="track-container">
      <Stage width={800} height={500} ref={stageRef}>
        <Layer>
          {/* Track outline */}
          <Line
            points={trackPoints}
            closed={true}
            stroke="#333"
            strokeWidth={15}
          />
          
          {/* Track border */}
          <Line
            points={trackPoints}
            closed={true}
            stroke="#aaa"
            strokeWidth={15}
            globalCompositeOperation="destination-over"
          />
          
          {/* Start/Finish line */}
          {trackCoordinates.length > 1 && (
            <Line
              points={calculateStartLine(trackCoordinates)}
              stroke="white"
              strokeWidth={5}
            />
          )}
          
          {/* Render drivers */}
          {renderDrivers()}
        </Layer>
      </Stage>
      
      {raceFinished && <RaceResults drivers={drivers} />}
    </div>
  );
};

// Helper to calculate start/finish line
const calculateStartLine = (coords) => {
  const startPoint = coords[0];
  const directionPoint = coords[1];
  const angle = Math.atan2(
    directionPoint[1] - startPoint[1], 
    directionPoint[0] - startPoint[0]
  ) + Math.PI / 2;
  
  const lineLength = 20;
  const startX1 = startPoint[0] + Math.cos(angle) * lineLength;
  const startY1 = startPoint[1] + Math.sin(angle) * lineLength;
  const startX2 = startPoint[0] - Math.cos(angle) * lineLength;
  const startY2 = startPoint[1] - Math.sin(angle) * lineLength;
  
  return [startX1, startY1, startX2, startY2];
};

export default TrackVisualizer;
