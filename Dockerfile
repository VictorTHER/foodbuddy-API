# Use desired python version as base
FROM python:3.10.6-buster

# First, pip install dependencies
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Only then copy taxifare!
COPY foodbuddy foodbuddy

# Command to launch the API web server
CMD uvicorn foodbuddy.api.fast:app --host 0.0.0.0 --port $PORT
