#!/bin/bash

sudo apt update
sudo apt install python3-pip

sudo apt install python3-picamera2

python3 -m venv --system-site-packages .env
source .env/bin/activate

pip3 install -r requirements.txt
