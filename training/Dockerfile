FROM tensorflow/tensorflow:latest-py3

WORKDIR /opt

COPY requirements.txt .
COPY training.py .

RUN pip3 install -r requirements.txt

CMD python3 training.py
