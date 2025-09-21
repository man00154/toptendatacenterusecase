# Use Python 3.13 slim image (CPU-only)
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip to avoid dependency issues
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy app source code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
