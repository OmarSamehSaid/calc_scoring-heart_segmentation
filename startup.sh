#!/bin/bash

# Start the MONAI Label server in the background
monailabel start_server --app radiology --studies . --conf models segmentation_unet_heart &

# Sleep briefly to give the server time to start
sleep 10

# Start your Flask application
python app.py
