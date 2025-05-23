# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install system dependencies and Python dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas3-base \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# Expose the Streamlit default port
EXPOSE 8501

# Run the Streamlit app when the container starts
CMD ["streamlit", "run", "app.py"]
