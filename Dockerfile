# Base image with Python and CUDA if GPU needed
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Streamlit run
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
