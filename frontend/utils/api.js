// Parameter ranges for standardization
const paramRanges = {
  pH: { min: 0, max: 14 },
  Hardness: { min: 0, max: 500 },
  Solids: { min: 0, max: 50000 },
  Chloramines: { min: 0, max: 10 },
  Sulfate: { min: 0, max: 500 },
  Conductivity: { min: 0, max: 1000 },
  Organic_carbon: { min: 0, max: 30 },
  Trihalomethanes: { min: 0, max: 200 },
  Turbidity: { min: 0, max: 10 }
};

export const standardizeValue = (value, min, max) => {
  // Scale to [-3, 3] range
  // Avoid division by zero if max equals min
  if (max === min) return 0; 
  return ((value - min) / (max - min)) * 6 - 3;
};

export const standardizeInput = (inputData) => {
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

export const predictWaterQuality = async (inputData) => {
  // Standardize the input data before sending
  const standardizedData = standardizeInput(inputData);
  
  console.log('Sending standardized data to API:', standardizedData);

  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    // Send the standardized data
    body: JSON.stringify(standardizedData),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.message || 'API request failed');
  }

  return response.json();
}; 