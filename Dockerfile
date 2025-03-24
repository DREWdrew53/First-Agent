FROM python:3.10.16

WORKDIR /aiserver

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "server.py"]