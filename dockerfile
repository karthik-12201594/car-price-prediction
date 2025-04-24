# Use Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install required OS dependencies and Python dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    statsmodels  # Added statsmodels

# Copy your files into the container
COPY "car_price_prediction.ipynb" ./car-price_prediction.ipynb
COPY "car_price_assignment.csv" ./car_Price_assignment.csv

# Expose port for Jupyter Notebook
EXPOSE 8888

# Run Jupyter Notebook when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
