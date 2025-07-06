# 1. Base image with torch preinstalled
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# 2. Set working directory
WORKDIR /app

# 3. System-level dependencies
RUN apt-get update && apt-get install -y \
    git \
    poppler-utils \
    libgl1-mesa-glx \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python dependencies
COPY requirements.txt .

# Force pip upgrade and no-cache installs
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir sentence-transformers rapidfuzz

# 5. Copy app code
COPY . .

# 6. Expose Streamlit default port
EXPOSE 80

# 7. Run app
CMD ["streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]
