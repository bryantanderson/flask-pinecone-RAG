import os
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec
from flask import render_template, request, jsonify
from flaskr import app
from flaskr.helpers import (
    generate_vectors,
    get_chat_completion,
    get_gemini_response,
    extract_text_from_file
)

load_dotenv()

pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = 'text-embedding-ada-002'
index_name = 'file-index'

if index_name not in [index['name'] for index in pinecone.list_indexes()]:
    pinecone.create_index(
        index_name, 
        dimension=1536,
        metric='cosine',
        spec=PodSpec(
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
    )

index = pinecone.Index(index_name)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat-gpt", methods=["POST"])
def get_bot_response():    
    user_query = request.json.get('user_input')

    # Create a vector using the user's message to compare similarity match using pinecone
    user_query_embedding = client.embeddings.create(input=[user_query], model=EMBEDDING_MODEL).data[0].embedding
    
    # Query pinecone to get similar vectors 
    similar_vectors = index.query(vector=user_query_embedding, top_k=1, include_metadata=True)
    
    # Extract the text associated with the embedding
    contexts = [item['metadata']['text'] for item in similar_vectors['matches']]
    
    # Add the extracted metadata text to use as additional context for the user's query
    augmented_user_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + user_query
    
    # Use this new augmented query with GPT
    gpt_response = get_chat_completion(augmented_user_query)  
    gemini_response = get_gemini_response(augmented_user_query)

    return jsonify({"gpt_message": gpt_response, "gemini_message": gemini_response})


@app.route("/files", methods=["POST"])
def process_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"})
    
    file = request.files['file'].read()
    file_docs = extract_text_from_file(file)
    # Extract the text content to convert into embeddings
    texts = [doc.page_content for doc in file_docs]
    file_vectors = generate_vectors(texts)

    # Get the ID of the last vector in the index
    index_stats = index.describe_index_stats()
    num_vectors = index_stats['total_vector_count']

    # Generate unique IDS for each vector
    ids = [str(index + 1) for index in range(num_vectors, num_vectors + len(file_vectors))]
    index.upsert(vectors=[(id, embedding, {"text": metadata}) for id, embedding, metadata in zip(ids, file_vectors, texts)])

    return "File upload successful"

