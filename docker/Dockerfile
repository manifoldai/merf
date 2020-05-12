FROM manifoldai/orbyter-ml-dev:2.0
ADD local_requirements.txt /build/local_requirements.txt
WORKDIR /build/
RUN pip install -r local_requirements.txt
WORKDIR /mnt/
