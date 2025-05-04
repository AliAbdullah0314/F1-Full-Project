import React, { useState, useEffect } from 'react';
import './SimulationConfigForm.css';

const SimulationConfigForm = ({ onSubmit, onTrackChange, selectedTrack }) => {
  // Track to round mapping for 2024
  const trackToRound = {
    'bahrain': '4',
    'austin': '19',
    'abu_dhabi': '24'
  };
  const trackToRound2024 = {
    'bahrain': '1',
    'austin': '19',
    'abu_dhabi': '24'
  };

  // Main form state
  const [formData, setFormData] = useState({
    year: '2024',
    round: selectedTrack || 'bahrain',
    model: 'LSTM',
    simulationType: 'default',
    safetyCarLaps: [],
    driverPitstops: {},
    driverCompounds: {},
    qualifyingData: {}
  });

  // Form step state
  const [currentStep, setCurrentStep] = useState(1);
  const [loading, setLoading] = useState(false);

  const [safetyCarText, setSafetyCarText] = useState('');
  const [pitstopTextInputs, setPitstopTextInputs] = useState({});

  // Driver data
  const [drivers, setDrivers] = useState([]);

  // Load drivers based on selected year
  useEffect(() => {
    const fetchDrivers = async () => {
      // For 2024, require both year and round; for 2025, only year
      if (formData.year === '2024' && !formData.round) {
        return; // Don't fetch until race is selected for 2024
      }

      try {
        // Construct API URL based on year and round
        const apiUrl = formData.year === '2024'
          ? `${process.env.REACT_APP_API_BASE_URL}/api/drivers/${formData.year}/${trackToRound2024[formData.round]}`
          : `${process.env.REACT_APP_API_BASE_URL}/api/drivers/${formData.year}`;

        const response = await fetch(apiUrl);
        const data = await response.json();
        setDrivers(data);
      } catch (error) {
        console.error("Error loading drivers:", error);
        // Fallback to sample data
        setDrivers([
          { id: 1, name: 'Max Verstappen', team: 'Red Bull Racing' },
          { id: 11, name: 'Sergio Perez', team: 'Red Bull Racing' },
          { id: 16, name: 'Charles Leclerc', team: 'Ferrari' },
          { id: 55, name: 'Carlos Sainz', team: 'Ferrari' },
          { id: 44, name: 'Lewis Hamilton', team: 'Mercedes' },
          { id: 63, name: 'George Russell', team: 'Mercedes' }
        ]);
      }
    };

    fetchDrivers();
  }, [formData.year, formData.round]);

  // Handle input changes
  const handleChange = (e) => {
    // e.preventDefault(); // Add this line
    const { name, value } = e.target;
    // // If year changes to 2025, clear round selection
    // if (name === 'year' && value === '2025') {
    //   setFormData({
    //     ...formData,
    //     [name]: value,
    //     round: ''
    //   });
    // }
    // else 
    // if (name === 'round' && onTrackChange) {
    //   onTrackChange(value);
    // } 
    // else {
    //   setFormData({
    //     ...formData,
    //     [name]: value
    //   });
    // }


    if (name === 'round' && onTrackChange) {
      onTrackChange(value);
    }
    setFormData({
      ...formData,
      [name]: value
    });


  };

  // // Handle safety car input
  // const handleSafetyCarChange = (e) => {
  //   const laps = e.target.value
  //     .split(',')
  //     .map(lap => parseInt(lap.trim()))
  //     .filter(lap => !isNaN(lap));

  //   setFormData({
  //     ...formData,
  //     safetyCarLaps: laps
  //   });
  // };

  // Modified handler for typing experience
  const handleSafetyCarChange = (e) => {
    // Allow raw text input
    setSafetyCarText(e.target.value);
  };

  // Add this blur handler to parse values
  const handleSafetyCarBlur = () => {
    const laps = safetyCarText
      .split(',')
      .map(lap => parseInt(lap.trim()))
      .filter(lap => !isNaN(lap));

    setFormData({
      ...formData,
      safetyCarLaps: laps
    });
  };

  // Handle pitstop changes for a driver
  const handlePitstopChange = (driverId, value) => {
    const pitstops = value
      .split(',')
      .map(lap => parseInt(lap.trim()))
      .filter(lap => !isNaN(lap));

    setFormData({
      ...formData,
      driverPitstops: {
        ...formData.driverPitstops,
        [driverId]: pitstops
      }
    });

    // Initialize compound data structure if needed
    if (!formData.driverCompounds[driverId]) {
      const compoundData = {};
      pitstops.forEach(lap => {
        compoundData[lap] = 1; // Default compound
      });

      setFormData(prev => ({
        ...prev,
        driverCompounds: {
          ...prev.driverCompounds,
          [driverId]: compoundData
        }
      }));
    }
  };

  const handlePitstopTextChange = (driverId, text) => {
    // Store raw text without parsing
    setPitstopTextInputs({
      ...pitstopTextInputs,
      [driverId]: text
    });
  };

  const handlePitstopBlur = (driverId, text) => {
    // Parse to numbers only when user has finished typing
    const pitstops = text
      .split(',')
      .map(lap => parseInt(lap.trim()))
      .filter(lap => !isNaN(lap));

    setFormData({
      ...formData,
      driverPitstops: {
        ...formData.driverPitstops,
        [driverId]: pitstops
      }
    });
  };


  useEffect(() => {
    setSafetyCarText(formData.safetyCarLaps.join(', '));
  }, []);

  // Initialize the text inputs when driver data changes
  useEffect(() => {
    const initialTexts = {};
    drivers.forEach(driver => {
      if (formData.driverPitstops[driver.id]) {
        initialTexts[driver.id] = formData.driverPitstops[driver.id].join(', ');
      }
    });
    setPitstopTextInputs(initialTexts);
  }, [drivers]);

  // Handle compound changes for a driver's pitstop
  const handleCompoundChange = (driverId, lap, compound) => {
    setFormData({
      ...formData,
      driverCompounds: {
        ...formData.driverCompounds,
        [driverId]: {
          ...formData.driverCompounds[driverId],
          [lap]: parseInt(compound)
        }
      }
    });
  };

  const handleGridPositionChange = (driverId, gridPosition) => {
    const position = parseInt(gridPosition);
    if (isNaN(position) || position < 1 || position > 20) return;

    setFormData({
      ...formData,
      qualifyingData: {
        ...formData.qualifyingData,
        [driverId]: {
          ...(formData.qualifyingData[driverId] || {}),
          grid: position
        }
      }
    });
  };

  // Handle qualifying time change
  const handleQualifyingTimeChange = (driverId, session, timeStr) => {
    const time = parseFloat(timeStr);
    if (isNaN(time)) return;

    setFormData({
      ...formData,
      qualifyingData: {
        ...formData.qualifyingData,
        [driverId]: {
          ...(formData.qualifyingData[driverId] || {}),
          [`${session}milli`]: time
        }
      }
    });
  };

  // Move to the next step
  const nextStep = (e) => {
    if (e) e.preventDefault();
    setCurrentStep(currentStep + 1);
  };

  // Move to the previous step
  const prevStep = () => {
    setCurrentStep(currentStep - 1);
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    // if (currentStep !== 5) {
    //   console.log("Prevented early submission - not on final step");
    //   return;
    // }
    setLoading(true);

    // Format the data for the backend
    const formattedData = {
      year: formData.year,
      round: formData.year == 2024 ? trackToRound2024[formData.round] : trackToRound[formData.round],
      model: formData.model,
      simulation_type: formData.simulationType
    };

    // Only include custom data if custom simulation is selected
    if (formData.simulationType === 'custom') {
      formattedData.custom_safety_car_laps = formData.safetyCarLaps;
      formattedData.custom_pitstops = formData.driverPitstops;
      formattedData.custom_compounds = formData.driverCompounds;
      formattedData.qualifying_data = formData.qualifyingData;
    }

    try {
      console.log('form submit')
      await onSubmit(formattedData);
      // Reset form or navigate away
    } catch (error) {
      console.error("Error submitting simulation:", error);
    } finally {
      setLoading(false);
    }
  };

  // Render different steps based on currentStep
  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="form-step">
            <h3>Step 1: Select Race Year</h3>
            <div className="input-group">
              <label>
                <input
                  type="radio"
                  name="year"
                  value="2024"
                  checked={formData.year === '2024'}
                  onChange={handleChange}
                />
                2024 Season
              </label>
              <label>
                <input
                  type="radio"
                  name="year"
                  value="2025"
                  checked={formData.year === '2025'}
                  onChange={handleChange}
                />
                2025 Season
              </label>
            </div>
            <div className="button-group">
              <button type="button" onClick={(e) => nextStep(e)}>Next</button>
            </div>
          </div>
        );

      case 2:
        return (
          <div className="form-step">
            <h3>Step 2: Select Race Track</h3>
            <div className="input-group">
              <label>
                <input
                  type="radio"
                  name="round"
                  value="bahrain"
                  checked={formData.round === 'bahrain'}
                  onChange={handleChange}
                />
                Bahrain Grand Prix
              </label>
              <label>
                <input
                  type="radio"
                  name="round"
                  value="austin"
                  checked={formData.round === 'austin'}
                  onChange={handleChange}
                />
                Austin Grand Prix
              </label>
              <label>
                <input
                  type="radio"
                  name="round"
                  value="abu_dhabi"
                  checked={formData.round === 'abu_dhabi'}
                  onChange={handleChange}
                />
                Abu Dhabi Grand Prix
              </label>
              {formData.year === '2025' && (
                <p className="note">Note: Track selection is recorded for 2025 simulations but doesn't affect driver data.</p>
              )}
            </div>
            <div className="button-group">
              <button type="button" onClick={prevStep}>Back</button>
              <button type="button" onClick={(e) => nextStep(e)}>Next</button>
            </div>
          </div>
        );

      case 3: // Step number updated
        return (
          <div className="form-step">
            <h3>Step 3: Select Prediction Model</h3>
            <div className="input-group">
              <label>
                <input
                  type="radio"
                  name="model"
                  value="LSTM"
                  checked={formData.model === 'LSTM'}
                  onChange={handleChange}
                />
                LSTM Model
              </label>
              <label>
                <input
                  type="radio"
                  name="model"
                  value="TFT"
                  checked={formData.model === 'TFT'}
                  onChange={handleChange}
                />
                TFT Model
              </label>
            </div>
            <div className="button-group">
              <button type="button" onClick={prevStep}>Back</button>
              <button type="button" onClick={(e) => nextStep(e)}>Next</button>
            </div>
          </div>
        );

      case 4:
        return (
          <div className="form-step">
            <h3>Step 4: Simulation Type</h3>
            <div className="input-group">
              <label>
                <input
                  type="radio"
                  name="simulationType"
                  value="default"
                  checked={formData.simulationType === 'default'}
                  onChange={handleChange}
                />
                Default Simulation
              </label>
              <label>
                <input
                  type="radio"
                  name="simulationType"
                  value="custom"
                  checked={formData.simulationType === 'custom'}
                  onChange={handleChange}
                />
                Custom Simulation
              </label>
            </div>
            <div className="button-group">
              <button type="button" onClick={prevStep}>Back</button>
              {formData.simulationType === 'default' ? (
                <button type="submit" disabled={loading}>
                  {loading ? 'Loading...' : 'Start Simulation'}
                </button>
              ) : (
                <button type="button" onClick={(e) => nextStep(e)}>Next</button>
              )}
            </div>
          </div>
        );

      case 5:
        return (
          <div className="form-step">
            <h3>Step 5: Custom Simulation Settings</h3>

            {/* Safety Car Input */}
            <div className="input-section">
              <h4>Safety Car Deployment Laps</h4>
              <div className="input-group">
                <label>
                  Enter lap numbers, separated by commas:
                  {/* <input
                    type="text"
                    placeholder="e.g., 5, 20, 35"
                    value={formData.safetyCarLaps.join(', ')}
                    onChange={handleSafetyCarChange}
                  /> */}

                  <input
                    type="text"
                    placeholder="e.g., 5, 20, 35"
                    value={safetyCarText}
                    onChange={handleSafetyCarChange}
                    onBlur={handleSafetyCarBlur}
                  />
                </label>
              </div>
            </div>

            {/* Driver Pitstops */}
            <div className="input-section">
              <h4>Driver Pitstops</h4>
              {drivers.map(driver => (
                <div key={driver.id} className="driver-input">
                  <h5>{driver.name} ({driver.team})</h5>
                  <div className="input-group">
                    <label>
                      Pitstop laps (comma separated):
                      {/* <input
                        type="text"
                        placeholder="e.g., 15, 35"
                        value={(formData.driverPitstops[driver.id] || []).join(', ')}
                        onChange={(e) => handlePitstopChange(driver.id, e.target.value)}
                      /> */}
                      <input
                        type="text"
                        placeholder="e.g., 15, 35"
                        value={pitstopTextInputs[driver.id] || ''}
                        onChange={(e) => handlePitstopTextChange(driver.id, e.target.value)}
                        onBlur={(e) => handlePitstopBlur(driver.id, e.target.value)}
                      />
                    </label>
                  </div>

                  {/* Compound Selection for each pitstop */}
                  {(formData.driverPitstops[driver.id] || []).length > 0 && (
                    <div className="compounds-input">
                      <h6>Tire Compounds:</h6>
                      {(formData.driverPitstops[driver.id] || []).map(lap => (
                        <div key={lap} className="compound-selector">
                          <label>
                            Lap {lap}:
                            <select
                              value={formData.driverCompounds[driver.id]?.[lap] ?? 1}
                              onChange={(e) => handleCompoundChange(driver.id, lap, e.target.value)}
                            >
                              <option value={0}>Soft (0)</option>
                              <option value={1}>Medium (1)</option>
                              <option value={2}>Hard (2)</option>
                            </select>
                          </label>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Qualifying Data Section */}
            <div className="input-section">
              <h4>Qualifying Data</h4>
              {drivers.map(driver => (
                <div key={`quali-${driver.id}`} className="driver-input">
                  <h5>{driver.name} ({driver.team})</h5>
                  <div className="quali-inputs">
                    <div className="input-group">
                      <label>
                        Grid Position (1-20):
                        <input
                          type="number"
                          min="1"
                          max="20"
                          value={formData.qualifyingData[driver.id]?.grid || ''}
                          onChange={(e) => handleGridPositionChange(driver.id, e.target.value)}
                        />
                      </label>
                    </div>
                    <div className="input-group">
                      <label>
                        Q1 Time (seconds):
                        <input
                          type="number"
                          step="0.001"
                          value={formData.qualifyingData[driver.id]?.q1milli || ''}
                          onChange={(e) => handleQualifyingTimeChange(driver.id, 'q1', e.target.value)}
                        />
                      </label>
                    </div>
                    <div className="input-group">
                      <label>
                        Q2 Time (seconds):
                        <input
                          type="number"
                          step="0.001"
                          value={formData.qualifyingData[driver.id]?.q2milli || ''}
                          onChange={(e) => handleQualifyingTimeChange(driver.id, 'q2', e.target.value)}
                        />
                      </label>
                    </div>
                    <div className="input-group">
                      <label>
                        Q3 Time (seconds):
                        <input
                          type="number"
                          step="0.001"
                          value={formData.qualifyingData[driver.id]?.q3milli || ''}
                          onChange={(e) => handleQualifyingTimeChange(driver.id, 'q3', e.target.value)}
                        />
                      </label>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="button-group">
              <button type="button" onClick={prevStep}>Back</button>
              <button type="submit" disabled={loading}>
                {loading ? 'Loading...' : 'Start Simulation'}
              </button>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="simulation-config-form">
      <h2>Race Simulation Configuration</h2>
      <form onSubmit={handleSubmit}>
        {renderStep()}
      </form>
    </div>
  );
};

export default SimulationConfigForm;
