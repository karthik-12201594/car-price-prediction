# Use Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install required OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy your files into the container
COPY Car\ Price\ Prediction.ipynb ./Car_Price_Prediction.ipynb
COPY CarPrice_Assignment\ \(1\).csv ./CarPrice_Assignment.csv

# Install Python dependencies
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    statsmodels  # Added statsmodels

# Expose port for Jupyter Notebook
EXPOSE 8888

# Run Jupyter Notebook when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
