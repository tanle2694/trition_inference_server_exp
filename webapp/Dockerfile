FROM python:3.8
WORKDIR /tmp
COPY . /tmp/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install tritonclient[all]==2.6.0
#CMD ["python", "flask_server.py"]