# 1. Use a lightweight official Python image
FROM python:3.11

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy just the requirements first to maximize Docker caching
COPY requirements.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your source code into the container
COPY src/ ./src/

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Start Uvicorn when the container boots up
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]