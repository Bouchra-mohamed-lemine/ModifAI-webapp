#!/usr/bin/env bash
set -o errexit

# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    wget \
    unzip \
    google-chrome-stable \
    libgl1-mesa-glx

# Install Chromedriver (match Chrome version)
CHROME_VERSION=$(google-chrome --version | awk '{print $3}')
CHROMEDRIVER_VERSION="${CHROME_VERSION%.*}"
wget -N "https://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip"
unzip chromedriver_linux64.zip -d ~/
sudo mv ~/chromedriver /usr/bin/chromedriver
sudo chmod +x /usr/bin/chromedriver

# Install Python dependencies
pip install -r requirements.txt