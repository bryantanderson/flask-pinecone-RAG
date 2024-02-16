from setuptools import setup

setup(
    name='flask_app',
    packages=['flask_app'],
    include_package_data=True,
    install_requires=[
        'flask',
        'openai',
        'google-generativeai',
        'pinecone-client',
        'langchain',
        'python-dotenv'
        'pypdf2',
        'gunicorn',
        'tiktoken',
        'langchain-openai'
    ],
)