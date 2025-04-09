FROM python:3.9-slim

# Install dependencies (Chrome + Chromedriver + system tools)
RUN apt-get update && apt-get install -y \
    chromium chromium-driver wget curl unzip gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for Chrome and Chromedriver
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Set the working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Start your app
CMD ["python", "app.py"]
