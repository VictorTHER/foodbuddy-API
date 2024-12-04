# Use desired python version as base
FROM python:3.10.6-buster

# First, pip install dependencies
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the entire project (including `api` and `foodbuddy`)
COPY . .

# Command to launch the API web server
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
