import React, { useState } from 'react';
import TrackVisualizer from './components/TrackVisualizer';
import StandingsTable from './components/StandingsTable';
import TrackSelector from './components/RaceControls/TrackSelector';
import PlaybackControls from './components/RaceControls/PlaybackControls';
import LapControls from './components/RaceControls/LapControls';
import SpeedControl from './components/RaceControls/SpeedControl';
import useRaceSimulation from './hooks/useRaceSimulation';
import useTrackData from './hooks/useTrackData';

import SimulationConfigForm from './components/SimulationConfigForm';

import './App.css';
import laptimescsv2024 from './data/laptimes/test_laptimes.csv';
import laptimescsv2025 from './data/laptimes/test_laptimes_2025.csv';

function App() {
  const [selectedTrack, setSelectedTrack] = useState('bahrain');
  const { trackCoordinates, totalLaps, loadTrack } = useTrackData();
  const [selectedYear, setSelectedYear] = useState("2024");
  const {
    drivers,
    isPlaying,
    speedMultiplier,
    currentVirtualTime,
    totalRaceTime,
    raceFinished,
    loadLapTimes,
    togglePlayback,
    updateSpeed,
    seekToLap,
    resetRace
  } = useRaceSimulation(trackCoordinates);

  const handleTrackChange = (trackName) => {
    setSelectedTrack(trackName);
    loadTrack(trackName);
  };
  const handleYearChange = (e) => {
    setSelectedYear(e.target.value);
  };
  const csvFiles = {
    "2024": laptimescsv2024,
    "2025": laptimescsv2025
  };

  const handleSimulationSubmit = async (simulationData) => {
    console.log("Simulation configuration:", simulationData);

    // Call your API to run the simulation
    try {
      const response = await fetch('http://localhost:8000/api/predict/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(simulationData),
      });

      const data = await response.json();

      // Load the simulation results
      loadLapTimes(data);
    } catch (error) {
      console.error("Error running simulation:", error);
    }
  };

  return (
    <div className="app-container">
      <h1>F1 Race Visualizer</h1>

      <SimulationConfigForm
        onSubmit={handleSimulationSubmit}
        onTrackChange={(track) => {
          setSelectedTrack(track);
          loadTrack(track);
        }}
        selectedTrack={selectedTrack}
      />
      <br></br>
      <div className="race-display-container">
        <StandingsTable drivers={drivers} />

        <TrackVisualizer
          trackCoordinates={trackCoordinates}
          drivers={drivers}
          isPlaying={isPlaying}
          raceFinished={raceFinished}
        />
      </div>

      <div className="controls-container">
        {/* <TrackSelector
          selectedTrack={selectedTrack}
          onTrackChange={handleTrackChange}
        /> */}

        {/* <h3>Select Race Data Year</h3>
        <div>
          <label>
            <input
              type="radio"
              name="year"
              value="2024"
              checked={selectedYear === "2024"}
              onChange={handleYearChange}
            />
            2024 Season
          </label>
          <label style={{ marginLeft: "20px" }}>
            <input
              type="radio"
              name="year"
              value="2025"
              checked={selectedYear === "2025"}
              onChange={handleYearChange}
            />
            2025 Season
          </label>
        </div>
        <button onClick={() => loadLapTimes(csvFiles[selectedYear])}>
          Load Lap Times from CSV
        </button> */}

        <PlaybackControls
          isPlaying={isPlaying}
          onTogglePlay={togglePlayback}
          progress={currentVirtualTime / totalRaceTime * 100}
          totalLaps={totalLaps}
          onSeek={(percent) => seekToLap((percent / 100) * totalLaps)}
          onReset={resetRace}
        />

        <LapControls
          totalLaps={drivers[0]?.lap_times?.length || 0}
          onSeekToLap={seekToLap}
        />

        <SpeedControl
          speed={speedMultiplier}
          onSpeedChange={updateSpeed}
        />
      </div>
    </div>
  );
}

export default App;
