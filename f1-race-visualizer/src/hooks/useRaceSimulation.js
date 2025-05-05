import { useState, useEffect, useRef } from 'react';

const useRaceSimulation = (trackCoordinates) => {
    // State variables
    const [drivers, setDrivers] = useState([]);
    const [isPlaying, setIsPlaying] = useState(false);
    const [speedMultiplier, setSpeedMultiplier] = useState(1);
    const [currentVirtualTime, setCurrentVirtualTime] = useState(0);
    const [totalRaceTime, setTotalRaceTime] = useState(0);
    const [raceFinished, setRaceFinished] = useState(false);
    const [totalLaps, setTotalLaps] = useState(3);
    const [lapChapters, setLapChapters] = useState([]);
    const [historicalMode, setHistoricalMode] = useState(false);

    // Refs for animation
    const animationRef = useRef(null);
    const lastTimestampRef = useRef(null);

    // Load lap times from CSV
    // Load lap times from backend JSON response
    const loadLapTimes = async (responseData) => {
        try {
            // Check if we're dealing with a file path or direct JSON data
            let data;
            if (typeof responseData === 'string') {
                // It's a file path, fetch it
                console.log('reading file ' + responseData)
                const response = await fetch(responseData);
                data = await response.json();
                console.log('after reading file ' + data)
            } else {
                // It's already JSON data
                data = responseData;
                console.log('received ' + data)
            }

            if (!data.success || !data.results) {
                throw new Error("Invalid data format or simulation unsuccessful");
            }

            const year = data.metadata.year
            const selected_round = data.metadata.selected_round

            const apiUrl = year === 2024
                ? `${window.configs.API_BASE_URL}/api/drivers/${year}/${selected_round}`
                : `${window.configs.API_BASE_URL}/api/drivers/${year}`;

            const response = await fetch(apiUrl);
            const driversJson = await response.json();


            // Extract driver lap times from results
            const results = data.results;
            const driverNames = Object.keys(results);

            // Find the maximum lap count across all drivers
            const maxLaps = Math.max(...driverNames.map(name => results[name].length));
            setTotalLaps(maxLaps);

            // Create lap chapters array
            const chapters = Array(maxLaps).fill().map((_, i) => i + 1);
            setLapChapters(chapters);


            const driverInfoMap = {};
            driversJson.forEach(driver => {
                // Normalize the name to match format in results (lowercase, replace spaces with underscores)
                const normalizedName = driver.name.toLowerCase().replace(/\s+/g, '_');
                driverInfoMap[normalizedName] = driver;
            });

            // Process driver data
            const newDrivers = [];
            driverNames.forEach((driverName, index) => {
                const lapTimes = results[driverName];

                const driverInfo = driverInfoMap[driverName] ||
                    // Fallback in case the normalization doesn't match exactly
                    driversJson.find(d => d.name.toLowerCase() === driverName.replace(/_/g, ' '));

                if (!driverInfo) {
                    console.warn(`Could not find driver info for ${driverName}`);
                    return; // Skip this driver
                }


                if (lapTimes && lapTimes.length > 0) {
                    newDrivers.push({
                        name: driverInfo.name,
                        color: teamColors[driverInfo.team] || '#666666',
                        lap_times: lapTimes,
                        current_lap: 0,
                        current_time: 0,
                        start_time: null,
                        virtual_time: 0,
                        lastProgress: 0,
                        completed: false,
                        totalLaps: 0
                    });
                }
            });

            setDrivers(newDrivers);

            // Calculate total race time based on first driver
            if (newDrivers.length > 0) {
                const totalTime = newDrivers[0].lap_times.reduce((a, b) => a + b, 0) * 1000;
                setTotalRaceTime(totalTime);
                console.log(`Total race time: ${totalTime / 1000} seconds`);
            }

            resetRace();
        } catch (error) {
            console.error("Error loading lap times:", error);
        }
    };

    // Animation logic
    useEffect(() => {
        if (!isPlaying || raceFinished || drivers.length === 0 || historicalMode) return;

        const animate = (timestamp) => {
            if (!lastTimestampRef.current) {
                lastTimestampRef.current = timestamp;
            }

            const delta = (timestamp - lastTimestampRef.current) * speedMultiplier;
            const newVirtualTime = currentVirtualTime + delta;

            setCurrentVirtualTime(newVirtualTime);

            // Calculate progress for progress bar
            const progress = totalRaceTime > 0 ?
                Math.min(100, (newVirtualTime / totalRaceTime) * 100) : 0;

            // Update driver positions
            updateDriverPositions(newVirtualTime);

            // Update timestamp for next frame
            lastTimestampRef.current = timestamp;

            // Continue animation if race isn't finished
            if (!raceFinished) {
                animationRef.current = requestAnimationFrame(animate);
            }
        };

        animationRef.current = requestAnimationFrame(animate);

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [isPlaying, raceFinished, speedMultiplier, drivers, currentVirtualTime, historicalMode, totalRaceTime]);

    // Update driver positions based on virtual time
    // Update driver positions based on virtual time
    const updateDriverPositions = (targetTime) => {
        if (!drivers.length) return;

        // Get the leader before updating positions
        const leader = [...drivers].sort((a, b) => {
            const aProgress = (a.current_lap || 0) + (a.lastProgress || 0);
            const bProgress = (b.current_lap || 0) + (b.lastProgress || 0);
            return bProgress - aProgress;
        })[0];

        let leaderFinishingRace = false;

        const updatedDrivers = drivers.map(driver => {
            // Skip update if driver already completed the race
            if (driver.completed) {
                return driver;
            }

            let elapsed = 0;
            let currentLap = 0;

            if (!driver.lap_times || driver.lap_times.length === 0) {
                return driver;
            }

            // Calculate which lap the driver is on
            while (currentLap < driver.lap_times.length) {
                const lapTime = driver.lap_times[currentLap] * 1000;
                if (elapsed + lapTime > targetTime) break;
                elapsed += lapTime;
                currentLap++;

                // Check if driver just completed a lap
                if (currentLap > (driver.totalLaps || 0)) {
                    // Update total laps tracker
                    driver = {
                        ...driver,
                        totalLaps: currentLap
                    };
                }

                if (currentLap >= driver.lap_times.length) {
                    currentLap = driver.lap_times.length - 1;
                    elapsed = targetTime;
                    break;
                }
            }

            // Calculate progress within current lap (0-1)
            const lapIndex = Math.min(currentLap, driver.lap_times.length - 1);
            const currentLapTime = driver.lap_times[lapIndex] * 1000;

            // Store previous progress to detect finish line crossing
            const previousProgress = driver.lastProgress || 0;

            const progress = currentLapTime > 0 ?
                Math.max(0, Math.min(1, (targetTime - elapsed) / currentLapTime)) : 0;

            // Check if this is the leader completing the final lap exactly
            if (driver.name === leader.name) {
                // console.log('checking ' + currentLap)
                // Check if driver is exactly at the final lap (not before, not after)
                if (currentLap === totalLaps - 1) {

                    // Check for crossing finish line (going from end of track to beginning)
                    if (progress > 0.99) {
                        console.log(`Leader ${driver.name} has completed exactly ${totalLaps} laps!`);
                        leaderFinishingRace = true;
                    }
                }
            }

            // Mark driver as having completed the race if the leader has finished
            let completed = driver.completed;
            if (leaderFinishingRace) {
                completed = true;
            }

            return {
                ...driver,
                current_lap: currentLap,
                current_time: targetTime,
                lastProgress: progress,
                completed: completed
            };
        });

        // Update drivers state
        setDrivers(updatedDrivers);

        // End race immediately when leader crosses finish line on the final lap
        if (leaderFinishingRace && !raceFinished) {
            console.log(`Race complete! Leader has finished all ${totalLaps} laps.`);
            setRaceFinished(true);
            setIsPlaying(false);

            // Force an immediate render to show race results
            setTimeout(() => {
                // This empty timeout ensures state updates are processed
            }, 0);
        }
    };



    // Toggle play/pause
    const togglePlayback = () => {
        if (historicalMode) {
            setHistoricalMode(false);
        }

        setIsPlaying(prev => !prev);
        lastTimestampRef.current = null;
    };

    // Update speed
    const updateSpeed = (newSpeed) => {
        setSpeedMultiplier(newSpeed);
    };

    // Seek to specific lap
    const seekToLap = (targetLap) => {
        if (!drivers.length) return;

        // console.log('targetLap:'+targetLap);

        try {
            // Pause playback first
            if (isPlaying) {
                setIsPlaying(false);
                if (animationRef.current) {
                    cancelAnimationFrame(animationRef.current);
                    animationRef.current = null;
                }
            }

            // Calculate time for this lap
            const { newTime, updatedDrivers } = calculateLapState(targetLap);

            // Update both states in one render cycle
            setCurrentVirtualTime(newTime);
            setDrivers(updatedDrivers);
            setHistoricalMode(true);
        } catch (e) {
            console.error("Error seeking to lap:", e);
        }
    };

    // Helper function to calculate all state at once
    const calculateLapState = (targetLap) => {
        // Find fastest driver to this lap
        const leader = drivers.reduce((a, b) => {
            const aTime = a.lap_times.slice(0, Math.min(targetLap, a.lap_times.length))
                .reduce((x, y) => x + y, 0);
            const bTime = b.lap_times.slice(0, Math.min(targetLap, b.lap_times.length))
                .reduce((x, y) => x + y, 0);
            return aTime < bTime ? a : b;
        });

        // Calculate target time
        const validTargetLap = Math.min(targetLap, leader.lap_times.length);
        const newTime = leader.lap_times.slice(0, validTargetLap)
            .reduce((a, b) => a + b, 0) * 1000;

        // Create new driver objects with updated positions
        const updatedDrivers = drivers.map(driver => {
            let elapsed = 0;
            let currentLap = 0;

            // Calculate lap position
            while (currentLap < driver.lap_times.length) {
                const lapTime = driver.lap_times[currentLap] * 1000;
                if (elapsed + lapTime > newTime) break;
                elapsed += lapTime;
                currentLap++;

                if (currentLap >= driver.lap_times.length) {
                    currentLap = driver.lap_times.length - 1;
                    elapsed = newTime;
                    break;
                }
            }

            // Calculate progress within current lap
            const lapIndex = Math.min(currentLap, driver.lap_times.length - 1);
            const currentLapTime = driver.lap_times[lapIndex] * 1000;
            const progress = currentLapTime > 0 ?
                Math.max(0, Math.min(1, (newTime - elapsed) / currentLapTime)) : 0;

            return {
                ...driver,
                current_lap: currentLap,
                current_time: newTime,
                lastProgress: progress
            };
        });

        return { newTime, updatedDrivers };
    };


    // Show historical lap (static view)
    const showHistoricalLap = (targetLap) => {
        seekToLap(targetLap);
        setHistoricalMode(true);
        setIsPlaying(false);
    };

    // Return to live view
    const returnToLive = () => {
        setHistoricalMode(false);
    };

    // Reset race
    const resetRace = () => {
        if (animationRef.current) {
            cancelAnimationFrame(animationRef.current);
        }

        lastTimestampRef.current = null;
        setCurrentVirtualTime(0);
        setRaceFinished(false);
        setHistoricalMode(false);

        // Reset driver data
        setDrivers(prev => prev.map(driver => ({
            ...driver,
            current_lap: 0,
            current_time: 0,
            start_time: null,
            virtual_time: 0,
            lastProgress: 0,
            completed: false,
            totalLaps: 0
        })));
    };

    // Get the current race leader
    const getLeader = () => {
        if (!drivers.length) return null;

        return [...drivers].sort((a, b) => {
            const aProgress = (a.current_lap || 0) + (a.lastProgress || 0);
            const bProgress = (b.current_lap || 0) + (b.lastProgress || 0);
            return bProgress - aProgress;
        })[0];
    };

    // Calculate race progress percentage
    const getRaceProgress = () => {
        if (totalRaceTime <= 0) return 0;
        return Math.min(100, (currentVirtualTime / totalRaceTime) * 100);
    };

    return {
        drivers,
        isPlaying,
        speedMultiplier,
        currentVirtualTime,
        totalRaceTime,
        raceFinished,
        totalLaps,
        lapChapters,
        historicalMode,
        raceProgress: getRaceProgress(),
        loadLapTimes,
        togglePlayback,
        updateSpeed,
        seekToLap,
        showHistoricalLap,
        returnToLive,
        resetRace,
        getLeader
    };
};

// Driver colors
const teamColors = {
    'Mercedes': '#27F4D2',       // Turquoise
    'Red Bull': '#3671C6',       // Blue
    'Ferrari': '#E80020',        // Red
    'McLaren': '#FF8000',        // Orange
    'Alpine F1 Team': '#FD4BC7', // Blue
    'RB F1 Team': '#6692FF',     // Light Blue
    'Aston Martin': '#229971',   // Green
    'Williams': '#64C4FF',       // Light Blue
    'Sauber': '#52E252',         // Green
    'Haas F1 Team': '#B6BABD'    // Silver/Gray
};


export default useRaceSimulation;
