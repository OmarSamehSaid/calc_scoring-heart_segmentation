# Use the official Python image as a builder stage
FROM python:3.9.13 AS builder
COPY . /app
# Set the working directory in the container
WORKDIR /app
# Install the Python dependencies
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Expose the port for Flask 
# Create a startup script that starts the MONAI Label server and then the Flask app
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Create a new stage for your actual application
FROM builder as app

# Expose port 5000 for your Flask application
EXPOSE 5000

# Specify the command to run your startup script
CMD ["/app/startup.sh"]