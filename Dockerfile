# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (required for some Python math packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch (CPU version to keep image lightweight) and other dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir sympy numpy streamlit

# Copy the rest of the application code and the weights
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the futuristic web UI
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]