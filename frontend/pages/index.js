import { useState } from 'react';

// Define parameter ranges here for sliders - updated to match FCM model inputs
const paramRanges = {
  "Max Fecal Coliform": { min: 0, max: 2000, step: 10 },
  "Min Temperature": { min: 5, max: 25, step: 0.1 },
  "Max BOD": { min: 0, max: 10, step: 0.1 },
  "Max Temperature": { min: 15, max: 40, step: 0.1 },
  "Max Conductivity": { min: 100, max: 1500, step: 10 },
  "Min Fecal Coliform": { min: 0, max: 1000, step: 10 },
  "Max Total Coliform": { min: 50, max: 3000, step: 10 },
  "Min BOD": { min: 0, max: 8, step: 0.1 },
  "Min Dissolved Oxygen": { min: 1, max: 10, step: 0.1 },
  "Min Nitrate N + Nitrite N": { min: 0, max: 6, step: 0.1 },
  "Min pH": { min: 4, max: 9, step: 0.1 },
  "Max Nitrate N + Nitrite N": { min: 0, max: 12, step: 0.1 }
};

// Function to get initial default values (e.g., midpoint)
const getInitialFormData = () => {
  const initialData = {};
  for (const param in paramRanges) {
    initialData[param] = ((paramRanges[param].max + paramRanges[param].min) / 2).toFixed(paramRanges[param].step < 1 ? 1 : 0);
  }
  return initialData;
};

// Standardization functions moved from utils/api.js
const standardizeValue = (value, min, max) => {
  // Scale to [-3, 3] range
  // Avoid division by zero if max equals min
  if (max === min) return 0; 
  return ((value - min) / (max - min)) * 6 - 3;
};

const standardizeInput = (inputData) => {
  const standardizedData = {};
  for (const [param, value] of Object.entries(inputData)) {
    const range = paramRanges[param];
    if (!range) {
        throw new Error(`Unknown parameter for standardization: ${param}`);
    }
    const floatValue = parseFloat(value);
    if (isNaN(floatValue)) {
      throw new Error(`Invalid input for ${param}: Please enter a number.`);
    }
    standardizedData[param] = standardizeValue(floatValue, range.min, range.max);
  }
  return standardizedData;
};

export default function Home() {
  const [formData, setFormData] = useState(getInitialFormData());
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isInfoModalOpen, setIsInfoModalOpen] = useState(false); // State for modal

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      // Ensure value is stored correctly, even if step is decimal
      [name]: parseFloat(value).toFixed(paramRanges[name].step < 1 ? 1 : 0)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setIsLoading(true);

    try {
      // No need to standardize - just send the original values
      console.log('Sending data to API:', formData);
      
      // Call our API
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || 'API request failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const getWaterColor = () => {
    if (!result) return 'bg-blue-200'; // Default color
    
    // Updated to match FCM 3-cluster output
    switch (result.cluster) {
      case 0: return 'bg-red-200';    // High Contamination (poor)
      case 1: return 'bg-yellow-200'; // Moderate Contamination
      case 2: return 'bg-green-200';  // Low Contamination (good)
      default: return 'bg-blue-200';  // Unknown/default
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-400 to-blue-600 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left side - Form */}
          <div className="bg-white/90 backdrop-blur-sm rounded-lg shadow-xl p-6">
            <h1 className="text-3xl font-bold mb-6 text-center text-blue-800">Water Quality Analysis</h1>
            
            <form onSubmit={handleSubmit} className="space-y-4">
              {Object.entries(formData).map(([param, value]) => {
                const range = paramRanges[param];
                return (
                  <div key={param} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <label className="block text-sm font-medium text-gray-800">
                        {param}
                      </label>
                      <span className="text-sm font-semibold text-blue-700 bg-blue-100 px-2 py-0.5 rounded">
                        {value} {/* Display current value */}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={range.min}
                      max={range.max}
                      step={range.step}
                      name={param}
                      value={value}
                      onChange={handleChange}
                      required
                      className="mt-1 block w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
                      // Style the thumb
                      style={{ 
                        background: `linear-gradient(to right, #60a5fa 0%, #60a5fa ${((value - range.min) / (range.max - range.min)) * 100}%, #e5e7eb ${((value - range.min) / (range.max - range.min)) * 100}%, #e5e7eb 100%)` 
                      }}
                    />
                  </div>
                );
              })}

              <button
                type="submit"
                disabled={isLoading}
                className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
                  isLoading ? 'bg-blue-400' : 'bg-blue-600 hover:bg-blue-700'
                } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
              >
                {isLoading ? 'Analyzing...' : 'Analyze Water Quality'}
              </button>
            </form>

            {error && (
              <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-md">
                Error: {error}
              </div>
            )}
          </div>

          {/* Right side - Flask and Results */}
          <div className="relative">
            <div className="bg-white/90 backdrop-blur-sm rounded-lg shadow-xl p-6 h-full flex items-center justify-center">
              <div className="flex flex-col items-center">
                {/* CSS-Based Chemical Flask */}
                <div className="relative w-64 h-64 mb-8">
                  <div className="absolute bottom-0 left-0 right-0">
                    <div className={`w-full h-48 rounded-b-full ${getWaterColor()} transition-colors duration-500`}></div>
                  </div>
                  <div className="absolute top-0 left-1/2 transform -translate-x-1/2">
                    <div className="w-32 h-10 bg-white/90 rounded-t-full"></div>
                  </div>
                  <div className="absolute top-10 left-1/2 transform -translate-x-1/2">
                    <div className="w-4 h-20 bg-white/90"></div>
                  </div>
                </div>

                {/* Results */}
                {result && (
                  <div className="text-center text-gray-800">
                    <h2 className="text-2xl font-semibold mb-2">{result.meaning}</h2>
                    <p className="text-gray-600">Cluster: {result.cluster}</p>
                    {result.certainty && (
                      <p className="text-gray-600">Certainty: {(result.certainty * 100).toFixed(1)}%</p>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Info Panel */}
        <div className="relative mt-8 bg-white/90 backdrop-blur-sm rounded-lg shadow-xl p-6">
          {/* Info Button */}
          <button 
            onClick={() => setIsInfoModalOpen(prev => !prev)} // Toggle modal
            className="absolute top-4 right-4 w-6 h-6 bg-blue-100 text-blue-700 rounded-full flex items-center justify-center text-sm font-bold hover:bg-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-500 z-10"
            aria-label="Show parameter explanations"
          >
            i
          </button>

           {/* Parameter Info Popup */}
          {isInfoModalOpen && (
            <div className="absolute top-4 right-4 w-80 bg-white rounded-lg shadow-xl z-20 p-4 border border-gray-200 transform -translate-y-full -translate-y-2">
              <button 
                onClick={() => setIsInfoModalOpen(false)}
                className="absolute top-1 right-1 w-6 h-6 text-gray-400 hover:text-gray-600 text-xl focus:outline-none"
                aria-label="Close parameter explanations"
              >
                &times;
              </button>
              <h3 className="text-md font-semibold mb-3 text-gray-800">Parameter Explanations:</h3>
              <div className="space-y-2 text-xs text-gray-700 max-h-64 overflow-y-auto pr-2">
                <p><strong className="font-medium text-gray-900">Max/Min Fecal Coliform:</strong> Bacteria indicating fecal contamination, measured in MPN/100mL.</p>
                <p><strong className="font-medium text-gray-900">Min/Max Temperature:</strong> Water temperature range, affects dissolved oxygen and microbial activity.</p>
                <p><strong className="font-medium text-gray-900">Min/Max BOD:</strong> Biochemical Oxygen Demand, measures organic pollution levels.</p>
                <p><strong className="font-medium text-gray-900">Max Conductivity:</strong> Ability of water to conduct electricity, indicates dissolved mineral content.</p>
                <p><strong className="font-medium text-gray-900">Max Total Coliform:</strong> Group of bacteria indicating potential contamination.</p>
                <p><strong className="font-medium text-gray-900">Min Dissolved Oxygen:</strong> Oxygen content in water, essential for aquatic life.</p>
                <p><strong className="font-medium text-gray-900">Min/Max Nitrate N + Nitrite N:</strong> Nitrogen compounds indicating agricultural runoff or sewage.</p>
                <p><strong className="font-medium text-gray-900">Min pH:</strong> Acidity or alkalinity of water, affects aquatic life and chemical reactions.</p>
              </div>
            </div>
          )}

          <h2 className="text-xl font-bold mb-4 text-blue-800">Water Quality Clusters</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-green-100 rounded-lg">
              <h3 className="font-semibold text-green-800">LOW CONTAMINATION (Cluster 2)</h3>
              <p className="text-sm text-gray-600">Good water quality with low contaminant levels, high dissolved oxygen, and balanced pH. Typically meets drinking water standards.</p>
            </div>
            <div className="p-4 bg-yellow-100 rounded-lg">
              <h3 className="font-semibold text-yellow-800">MODERATE CONTAMINATION (Cluster 1)</h3>
              <p className="text-sm text-gray-600">Acceptable water quality with moderate contaminant levels. May require standard treatment before human consumption.</p>
            </div>
            <div className="p-4 bg-red-100 rounded-lg">
              <h3 className="font-semibold text-red-800">HIGH CONTAMINATION (Cluster 0)</h3>
              <p className="text-sm text-gray-600">Poor water quality with elevated levels of contaminants, low dissolved oxygen, and potentially extreme pH values. Requires extensive treatment.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
