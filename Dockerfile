# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install build tools including g++ needed for packages like annoy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
# Using --default-timeout to avoid issues with slow package downloads
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Copy the rest of the application code into the container at /app
# This includes app.py, pipeline_scripts/, and any other necessary files/folders
# IMPORTANT: The 'data' directory is NOT copied here by default to keep image size manageable.
# Users should mount their data directory or copy data into the container post-build if needed.
# For the app to find data, it expects a 'data/h5ad_filt' structure if data is placed inside /app
COPY app.py .
COPY pipeline_scripts/ ./pipeline_scripts/
# If you have other assets like images or CSS that app.py uses directly (not data files), copy them too.
# COPY assets/ ./assets/

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=True
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=False


# Run app.py when the container launches
# Use Healthcheck to make sure that the application is running
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1
  
CMD ["streamlit", "run", "app.py"] 