FROM manifoldai/orbyter-ml-dev:2.0
ADD requirements.txt /build/requirements.txt
ADD nb_requirements.txt /build/nb_requirements.txt
WORKDIR /build/
RUN pip install -r requirements.txt
RUN pip install -r nb_requirements.txt
WORKDIR /mnt/
