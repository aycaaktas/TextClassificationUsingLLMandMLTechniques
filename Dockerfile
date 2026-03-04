# Step 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY . /app

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# To allow running different scripts via command line
CMD ["python", "models/ml_model.py"]  

# To run the other, you could override the command during `docker run`
# docker run hepsiburadacasestudy python models/llm_model.py

# Note: The llm_model.py script may not work as intended because it relies on accessing 
# the Ollama API and LLM model, which were used locally during development but could not 
# be bound to the Docker container. You may encounter issues with accessing these resources
# when running the script within the container.
