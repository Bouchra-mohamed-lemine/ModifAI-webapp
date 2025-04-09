#!/usr/bin/env bash
set -o errexit

# Set up local binaries
mkdir -p ~/.local/bin

# Install Chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
dpkg -x google-chrome-stable_current_amd64.deb /tmp/chrome
mv /tmp/chrome/opt/google/chrome/google-chrome ~/.local/bin/
rm google-chrome-stable_current_amd64.deb

# Install Chromedriver
CHROME_VERSION=$(~/.local/bin/google-chrome --version | awk '{print $3}')
CHROMEDRIVER_VERSION=$(curl -s "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_${CHROME_VERSION%.*}")
wget "https://chromedriver.storage.googleapis.com/${CHROMEDRIVER_VERSION}/chromedriver_linux64.zip"
unzip chromedriver_linux64.zip
mv chromedriver ~/.local/bin/
chmod +x ~/.local/bin/chromedriver

# Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Install Python dependencies
pip install -r requirements.txt
