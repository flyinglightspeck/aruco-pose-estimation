#!/bin/bash

sudo apt update
sudo apt install python3-pip

python3 -m venv .env
source .env/bin/activate

pip3 install -r requirements.txt
sudo apt install python3-picamera2
