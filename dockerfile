FROM python:3.9
COPY . /app
EXPOSE 8000
WORKDIR /app
RUN pip install -r requirements.txt
CMD uvicorn image:app --host 0.0.0.0 --port 8000
