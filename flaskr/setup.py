from setuptools import setup

setup(
    name='flaskr',
    packages=['flaskr'],
    include_package_data=True,
    install_requires=[
        'flask',
        'openai',
        'google-generativeai',
        'pinecone-client',
        'langchain',
        'python-dotenv'
        'pypdf2',
        'gunicorn'
    ],
)