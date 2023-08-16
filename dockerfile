FROM python:3.10.12-slim

RUN pip install -U pip
# RUN pip install pipenv 

# WORKDIR /app

# COPY [ "Pipfile", "Pipfile.lock", "./" ]

COPY . .

# RUN pipenv install --system --deploy

# COPY [ "predict.py", "lin_reg.bin", "./" ]

RUN pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:8000", "application:app" ]

# gunicorn --bind=0.0.0.0:8000 app:app

# sudo docker build -t mosesdaudu001/tumor-predictor .
# sudo docker run -it --rm -p 8000:8000 mosesdaudu001/tumor-predictor