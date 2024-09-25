export default async function handler(req, res) {
  try {
    // Forward the request to the backend server
    const backendRes = await fetch('http://localhost:8000/experiments/', {
      method: req.method, // Forward the HTTP method
      headers: {
        // Forward the relevant headers
        'Content-Type': 'application/json'
      },
      body: req.method !== 'GET' ? JSON.stringify(req.body) : undefined,
    });

    // Forward the backend response status code
    res.status(backendRes.status);

    // Forward the backend response body
    const data = await backendRes.json();
    res.json(data);
  } catch (error) {
    // Handle any errors
    res.status(500).json({ error: 'Error proxying to backend.' });
  }
}
