FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime


# Set environment variables
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /code

# Copy everything from src directory to /code in the container
COPY ./src /code

# Update package lists, add deadsnakes PPA, and install Python 3.8 and pip
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-distutils python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#RUN python3 /code/utils/load_model.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


