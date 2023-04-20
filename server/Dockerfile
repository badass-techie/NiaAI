# use this image as the base
FROM python:3.11.1-slim-buster

# set environment variables
# don't write .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# don't buffer python output
ENV PYTHONUNBUFFERED=1

# create a working directory in the container and cd into it
WORKDIR /server

# copy and install dependencies to working directory
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

# copy source code to working directory
COPY . .
