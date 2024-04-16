# Start your image with a node base image
FROM python:3.11

# The /app directory should act as the main application directory
WORKDIR /classification_model

COPY ./ ./

# Install pip in docker container
# RUN apt-get update && 
# RUN apt-get update && apt-get install
# RUN apt-get install -y python3
# RUN apt-get install -y python3-pip

RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt
# EXPOSE 3000

# Start the app using serve command
CMD [ "python", "model.py"]