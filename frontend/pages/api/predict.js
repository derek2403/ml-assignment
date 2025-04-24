// Next.js API route handler for water quality prediction
// This proxies requests to the external prediction server

export default async function handler(req, res) {
  // Only accept POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    // Forward the request to the external server
    const response = await fetch(
      'https://e3c329acf714051138becd9199470e6d1ae0cabd-5050.dstack-prod5.phala.network/predict',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(req.body),
      }
    );

    // Handle non-successful responses
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return res.status(response.status).json(
        errorData.message ? { message: errorData.message } : { message: 'API request failed' }
      );
    }

    // Return the successful response data
    const data = await response.json();
    return res.status(200).json(data);
  } catch (error) {
    console.error('Error proxying request to prediction server:', error);
    return res.status(500).json({ message: 'Failed to connect to prediction server' });
  }
} 