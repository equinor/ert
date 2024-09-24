import { useState, useEffect } from 'react';
import ProgressBar from '../components/ProgressBar';

export default function Home() {
  const [experiments, setExperiments] = useState([]);
  const [selectedExperimentId, setSelectedExperimentId] = useState('');
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    // Fetch the list of experiments
    const fetchExperiments = async () => {
      try {
        const res = await fetch('/api/experiments');
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        const data = await res.json();
        setExperiments(data);
        if (data.length > 0 && !selectedExperimentId) {
          setSelectedExperimentId(data[0].id);
        }
      } catch (error) {
        console.error("Failed to fetch experiments:", error);
      }
    };

    // Initial fetch
    fetchExperiments();

    // Set up polling
    const intervalId = setInterval(fetchExperiments, 5000); // Poll every 5 second

    // Clear interval on cleanup
    return () => clearInterval(intervalId);
  }, [selectedExperimentId]); // Only re-run if selectedExperimentId changes

  useEffect(() => {
    if (!selectedExperimentId) return;

    const wsUrl = `ws://127.0.0.1:8000/experiments/${selectedExperimentId}/events`;
    const socket = new WebSocket(wsUrl);

    socket.onmessage = function (event) {
      console.log('Message from WS Server', event.data);
      const messageData = JSON.parse(event.data);
      if (messageData.progress !== undefined) {
        const progressPercentage = messageData.progress * 100;
        setProgress(progressPercentage);
      }
    };

    socket.onopen = () => console.log('Connected to WS Server', wsUrl);
    socket.onclose = () => console.log('Disconnected from WS Server');
    socket.onerror = (error) => console.error('WebSocket Error:', error);

    return () => {
      socket.close();
    };
  }, [selectedExperimentId]); // Re-run this effect if the selected experiment ID changes

  const handleDropdownChange = (e) => {
    setSelectedExperimentId(e.target.value);
  };

  return (
    <div style={{ padding: '20px', maxWidth: '600px', margin: 'auto' }}>
      <h1>WebSocket Progress Bar Example</h1>
      <div>
        <label htmlFor="experiment-select">Choose an experiment:</label>
        <select
          id="experiment-select"
          value={selectedExperimentId}
          onChange={handleDropdownChange}
          disabled={experiments.length === 0}
        >
          {experiments.map((experiment) => (
            <option key={experiment.id} value={experiment.id}>
              {experiment.id}
            </option>
          ))}
        </select>
      </div>
      <ProgressBar progress={progress} />
    </div>
  );
}
