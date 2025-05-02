# Use Python 3.10 as the base image
FROM python:3.10-slim

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files, predict.py and data.csv
COPY predict.py ./
COPY final.csv ./
COPY *.joblib ./

# Expose the port your Flask app runs on
EXPOSE 5050

# Command to run the Flask server
CMD ["python", "predict.py"]