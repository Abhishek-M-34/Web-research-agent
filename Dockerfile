# Use an official lightweight Python runtime
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Create a non-root user (Hugging Face Spaces requirement for security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# Run the Flask app with Gunicorn on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
