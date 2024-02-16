To run the flask app, go through the following commands:
cd flaskr
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cd ..
gunicorn -w 4 -b 127.0.0.1:5000 flask_app:app

To run streamlit:
cd streamlit-app
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py