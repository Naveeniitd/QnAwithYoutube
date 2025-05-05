# Variables
IMAGE_NAME = youtube-qa-app
PORT = 8501

# Build Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the Docker container
run:
	docker run -p $(PORT):8501 $(IMAGE_NAME)

# Remove Docker image
clean:
	docker rmi -f $(IMAGE_NAME)

# Rebuild image from scratch
rebuild: clean build

# Run in detached mode
run-detached:
	docker run -d -p $(PORT):8501 --name $(IMAGE_NAME)-container $(IMAGE_NAME)

# Stop and remove container
stop:
	docker stop $(IMAGE_NAME)-container || true
	docker rm $(IMAGE_NAME)-container || true