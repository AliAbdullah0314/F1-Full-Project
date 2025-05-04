import { useState, useEffect } from 'react';
import { processTrackCoordinates } from '../utils/trackUtils';

const useTrackData = () => {
  const [trackCoordinates, setTrackCoordinates] = useState([]);
  const [currentTrack, setCurrentTrack] = useState('bahrain');
  const [totalLaps, setTotalLaps] = useState(57);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadTrack = async (trackName) => {
    try {
      setLoading(true);
      setError(null);
      
      // Dynamic import for track data
      const trackModule = await import(`../data/tracks/${trackName}.js`);
      const trajectory = trackModule.default || trackModule.trajectory;
      const totalLaps = trackModule.totalLaps
      
      if (!trajectory || !trajectory.length) {
        throw new Error(`No valid track data found for ${trackName}`);
      }
      
      // Process trajectory into coordinates
      const coordinates = processTrackCoordinates(trajectory);

      
      setTrackCoordinates(coordinates);
      setCurrentTrack(trackName);
      setTotalLaps(totalLaps)
      setLoading(false);
    } catch (err) {
      console.error(`Error loading track ${trackName}:`, err);
      setError(`Failed to load track data for ${trackName}`);
      setLoading(false);
    }
  };

  // Load initial track data
  useEffect(() => {
    loadTrack(currentTrack);
  }, []);

  return {
    trackCoordinates,
    currentTrack,
    totalLaps,
    loading,
    error,
    loadTrack
  };
};

export default useTrackData;
