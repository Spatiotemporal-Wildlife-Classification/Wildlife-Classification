# Use TensorFlow with GPU support as the base image
FROM tensorflow/tensorflow:2.9.1-gpu

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Set the default command to run when the container starts
CMD ["python", "/app/src/models/image/taxonomic/taxonomic_modelling.py"]  # Training docker image
#CMD ["python", "/app/src/models/image/taxonomic/evaluate_taxonomic_model.py"]  # Evaluation docker image
