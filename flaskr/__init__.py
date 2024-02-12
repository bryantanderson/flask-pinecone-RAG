import sqlite3
from flask import Flask

app = Flask(__name__)

def get_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect("database.db")
    connection.row_factory = sqlite3.Row
    return connection

import flaskr.views