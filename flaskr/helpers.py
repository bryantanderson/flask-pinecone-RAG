import os
import io
import PyPDF2
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel('gemini-pro')

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
    try:
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

    except Exception as e:
        print(f"Error while attempting to generate response using GPT: {e}")

def get_gemini_response(user_message):
    try:
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
    
    except Exception as e:
        print(f"Error while attempting to generate response using Gemini: {e}")

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

def generate_summary(text):
    try:
        """Takes a text and generates a summary using Gemini."""
        gemini_prompt = f"""
        You will be given a text, which will be after the # delimiter. 
        Please summarize the text in 500 words or less.
        ##############################################################
        {text}
        """
        model_response = gemini.generate_content(gemini_prompt).text
        formatted_model_message = {
            "role": "model",
            "parts": [model_response]
        }
        gemini_messages.append(formatted_model_message)
        return model_response
    
    except Exception as e:
        print(f"Error while trying to summarize using Gemini: {e}")

