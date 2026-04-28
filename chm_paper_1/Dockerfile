# Use a lightweight Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the rest of your application code into the container
COPY . .

WORKDIR /app/chm_paper_1

RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit uses
EXPOSE 8501

# Start the Streamlit app and bind it to all network interfaces (0.0.0.0)
CMD ["streamlit", "run", "circuit_matrices_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
