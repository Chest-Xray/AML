FROM python:3.12-slim

# Install system libraries required by OpenCV and GUI-related libs (libxcb etc.)
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		libglib2.0-0 \
		libsm6 \
		libxrender1 \
		libxext6 \
		libx11-6 \
		libxcb1 \
		libgl1 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3000

CMD ["python", "-m", "uvicorn", "chest_xray.interfaces.backend.api.api:app", "--reload", "--host", "0.0.0.0", "--port", "3000"]