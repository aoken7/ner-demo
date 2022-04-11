FROM pytorch/pytorch:latest
#1.9.1-cuda11.1-cudnn8-runtime

LABEL maintainer="YuichiTasho"

RUN apt-get update -y
RUN apt-get install -y tzdata 
RUN apt-get install -y 
RUN pip install pandas pytorch_lightning transformers fugashi ipadic seqeval \
    bertviz jupyterlab ipywidgets flask flask_cors Flask-SSLify
#CMD [ "python3", "../main.py" ]
CMD [ "python3","/working_dir/backend/main.py" ]