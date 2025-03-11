FROM tensorflow/tensorflow:2.9.1-gpu
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "AstalAlgorithmists_Assignment_3.2.py"]
