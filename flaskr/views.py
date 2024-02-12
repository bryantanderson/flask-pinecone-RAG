import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec
from flask import request, jsonify
from flaskr import app, get_db_connection
from flaskr.helpers import (
    generate_vectors,
    get_chat_completion,
    extract_text_from_file,
    generate_summary,
    generate_file_chunks,
)
from flaskr.rag import get_hypothetical_response_embedding

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

@app.route("/chat", methods=["POST"])
def get_bot_response():    
    try:        
        user_query = request.json.get('user_input')
        use_rag = request.json.get('rag')
        gpt_response = None
        print(f"Get bot response called with input {request}")
        if use_rag:
            # user_query_embedding = client.embeddings.create(input=[user_query], model=EMBEDDING_MODEL).data[0].embedding
            user_query_embedding = get_hypothetical_response_embedding(user_query)

            # Query pinecone to get similar vectors 
            similar_vectors = index.query(vector=user_query_embedding, top_k=3, include_metadata=True)

            # Extract the text associated with the embedding
            contexts = [item['metadata']['text'] for item in similar_vectors['matches']]

            # Add the extracted metadata text to use as additional context for the user's query
            context = "\n\n---\n\n".join(contexts)
            augmented_user_query = context + "\n\n-----\n\n" + user_query

            # Use this new augmented query with GPT
            gpt_response = get_chat_completion(augmented_user_query)  

        else:
            gpt_response = get_chat_completion(user_query) 

        return jsonify({"message": gpt_response, "error": False})
    
    except Exception as e:
        return jsonify({"message": f"Error while trying to generate a response: {e}", "error": True})


@app.route("/files", methods=["POST"])
def process_and_vectorize_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"})

        print(f"Process and vectorize file was called with input {request}")
        file = request.files['file'].read()
        file_text = extract_text_from_file(file)
        file_docs = generate_file_chunks(file_text)
        # Extract the text content from the chunks to convert into embeddings
        texts = [doc.page_content for doc in file_docs]
        file_vectors = generate_vectors(texts)

        # Delete all old indexes to prevent overlap
        index.delete(delete_all=True)

        # Get the ID of the last vector in the index
        index_stats = index.describe_index_stats()
        num_vectors = index_stats['total_vector_count']

        # Generate unique IDS for each vector
        ids = [str(index + 1) for index in range(num_vectors, num_vectors + len(file_vectors))]
        index.upsert(vectors=[(id, embedding, {"text": metadata}) for id, embedding, metadata in zip(ids, file_vectors, texts)])

        return jsonify({"message": f"File uploaded successfully!", "error": False})
    
    except Exception as e:
        return jsonify({"message": f"Error while trying to process File: {e}", "error": True})
    

@app.route("/summarize-file", methods=["POST"])
def summarize_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"})

        print(f"Summarize file called with input {request}")
        file = request.files['file']
        file_bytes = file.read()
        file_size = len(file_bytes)
        file_text = extract_text_from_file(file_bytes)
        generate_summary(file_text)


        return jsonify({"message": f"File processed successfully!", "error": False})
    
    except Exception as e:
        return jsonify({"message": f"Error while trying to process File: {e}", "error": True})


@app.route("/clear", methods=["POST"])
def clear_file_information():
    try:
        print(f"Clear file information called with input {request}")
        index.delete(delete_all=True)
        return jsonify({"message": f"File information deleted successfully!", "error": False})

    except Exception as e:
        return jsonify({"message": f"Error while trying to delete file information: {e}", "error": True})

