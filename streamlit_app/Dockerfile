FROM python:3.11

EXPOSE 8501

COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

COPY . .
WORKDIR /app

CMD ["python", "-m", "streamlit", "run", "app.py", "-server.port=8501", "-server.address=0.0.0.0"]