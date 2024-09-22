# Use the lightweight Python 3.10 slim image
FROM python:3.10-slim

# Set environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY freeze_file.txt ./

# Install Python dependencies from the freeze file
RUN pip install --no-cache-dir --verbose -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the command to run your application
CMD ["python", "./src/WAB_Cifar10.py"]
