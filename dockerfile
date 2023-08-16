FROM python:3.11.4

RUN pip install -U pip

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:8000", "application:app" ]

# gunicorn --bind=0.0.0.0:8000 app:app

# sudo docker build -t mosesdaudu001/tumor-predictor .
# sudo docker run -it --rm -p 8000:8000 mosesdaudu001/tumor-predictor