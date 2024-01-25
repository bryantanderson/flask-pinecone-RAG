To run the flask app, go through the following commands:

python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
gunicorn -w 4 -b 127.0.0.1:5000 flaskr:app
