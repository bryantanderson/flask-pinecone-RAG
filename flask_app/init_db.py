import sqlite3

connection = sqlite3.connect('database.db') 
with open('schema.sql') as file:
    connection.executescript(file.read())
# create a Cursor object to process rows in a database
db = connection.cursor()
connection.commit()
connection.close()

