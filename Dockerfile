# Use an official standard Python image (not slim) to prevent missing build dependencies
FROM python:3.10

# Create a non-root user (Hugging Face Spaces requirement for security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY --chown=user . .

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Run the Flask app with Gunicorn on port 7860, utilizing 2 workers
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "-w", "2", "app:app"]
