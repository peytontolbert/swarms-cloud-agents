# Dockerfile:c:\Users\Peyton\AppData\Local\Programs\Python\Python311\Lib\site-packages\swarms\structs\agent.py

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt


# Copy the application code
COPY . .

# Expose the necessary port (e.g., 8000)
EXPOSE 5000

# Define the default command to run the agent
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "4"]