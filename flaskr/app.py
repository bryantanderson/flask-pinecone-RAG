import os
import io
import PyPDF2
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec
from flask import Flask, render_template, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

app = Flask(__name__)

pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel('gemini-pro')

EMBEDDING_MODEL = 'text-embedding-ada-002'
index_name = 'file-index'

if index_name not in [index['name'] for index in pinecone.list_indexes()]:
    print(pinecone.list_indexes())
    pinecone.create_index(
        index_name, 
        dimension=1536,
        metric='cosine',
        spec=PodSpec(
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
    )

index = pinecone.Index(index_name)

gpt_messages = [
    {
        "role": "system", 
        "content": 
        """
        You are an intelligent AI assistant that is able to answer anything with great detail and passion. 
        You are helpful, friendly, and your mission is to help people with any queries they may have.
        To ensure a smooth user experience, please limit your responses to a maximum of 100 words.
        """
    }
]

gemini_messages = []

def get_chat_completion(user_message):
    new_messages = gpt_messages.copy()
    new_messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=new_messages,
        temperature=0, 
    )

    # Store assistant message
    assistant_message = response.choices[0].message
    gpt_messages.append(assistant_message)

    # Convert ChatCompletionMessage to dictionary
    assistant_message_dict = {
        "content": assistant_message.content,
        "role": assistant_message.role,
        # Add other attributes you want to include
    }

    return assistant_message_dict["content"]

def get_gemini_response(user_message):
    formatted_user_message = {
        "role": "user",
        "parts": [user_message]
    }
    gemini_messages.append(formatted_user_message)
    
    model_response = gemini.generate_content(gemini_messages).text

    formatted_model_message = {
        "role": "model",
        "parts": [model_response]
    }
    gemini_messages.append(formatted_model_message)

    return model_response

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


def extract_text_from_file(file):
    """Takes a PDF, extracts the text and converts it into a list of chunks"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file))
    numPages = len(pdf_reader.pages)
    text = ""
    for i in range(numPages):
        page = pdf_reader.pages[i]
        text += page.extract_text()

    text = text.replace("\t", " ")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[],
        chunk_size=1536,
        chunk_overlap=200,
    )
    
    docs = text_splitter.create_documents([text])
    return docs

def generate_vectors(texts):
    """Takes a list of texts, and converts them into embeddings"""
    embeddings_response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    embeddings = [data.embedding for data in embeddings_response.data]
    return embeddings

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

if __name__ == "__main__":
    app.run()


