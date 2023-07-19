# Use the official Python image from the Docker Hub
FROM python:3.11

# Make a directory for our application
WORKDIR /usr/src/app

# Install the requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Run Makefile commands
CMD ["make", "all"]