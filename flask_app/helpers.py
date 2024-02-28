import os
import io
import PyPDF2
import tiktoken
import google.generativeai as genai
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter, Document

load_dotenv()

pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel('gemini-pro')
token_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")

EMBEDDING_MODEL = 'text-embedding-ada-002'
CONTEXT_WINDOW = 16385

index_name = 'file-index'
summarization_index = None
total_token_usage = 0

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

gpt_chat_history = [
    {
        "role": "system", 
        "content": 
        """
        You are an intelligent AI assistant that is able to answer anything in great detail. 
        You are helpful, friendly, and your mission is to answer any queries a user may have.
        To ensure a smooth user experience, limit all your responses to a maximum of 300 words.
        When answering, ensure to include specific details, such as specific dates or names if given
        such information. 
        """
    }
]


def get_chat_completion(user_message: str) -> str:
    try:
        gpt_chat_history.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model='gpt-3.5-turbo-0125',
            messages=gpt_chat_history,
            temperature=0, 
        )

        # Store assistant message
        assistant_message = response.choices[0].message
        # Convert ChatCompletionMessage to dictionary
        assistant_message_dict = {
            "content": assistant_message.content,
            "role": assistant_message.role,
            # Add other attributes you want to include
        }
        gpt_chat_history.append(assistant_message_dict)

        return assistant_message.content

    except Exception as e:
        gpt_chat_history.pop()
        print(f"Error while attempting to generate response using GPT: {e}")


def extract_text_from_file(file) -> str:
    """Takes a PDF, extracts the text and converts it into a list of chunks"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file))
    numPages = len(pdf_reader.pages)
    text = ""
    for i in range(numPages):
        page = pdf_reader.pages[i]
        text += page.extract_text()

    text = text.replace("\t", " ")
    return text


def generate_file_chunks(
    text: str, 
    desired_chunk_size=256, 
    desired_chunk_overlap=25
) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[],
        chunk_size=desired_chunk_size,
        chunk_overlap=desired_chunk_overlap,
    )
    docs = text_splitter.create_documents([text])
    return docs


def generate_vectors(texts: List[str]) -> List[List[float]]:
    """Takes a list of texts, and converts them into embeddings"""
    embeddings_response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    embeddings = [data.embedding for data in embeddings_response.data]
    return embeddings


def generate_summary(text: str) -> None:
    try:
        """Takes a text and generates a summary using OpenAI."""

        prompt = f"""
        You will be given a text, which will be placed after the # delimiter. 
        Summarize the text, making sure to retain important information,
        such as the specific names of people, dates, and numbers.
        #######################################################################
        {text}
        """

        # Check that the prompt will not exceed the context window
        gpt_chat_history.append({
            "role": "user",
            "content": prompt
        })

        response = client.chat.completions.create(
            model='gpt-3.5-turbo-0125',
            messages=gpt_chat_history,
            temperature=0, 
        )

        # Store assistant message
        assistant_message = response.choices[0].message
        # Convert ChatCompletionMessage to dictionary
        assistant_message_dict = {
            "content": assistant_message.content,
            "role": assistant_message.role,
        }
        gpt_chat_history.append(assistant_message_dict)
    
    except Exception as e:
        gpt_chat_history.pop()
        print(f"Error while trying to summarize using GPT: {e}")


def get_num_vectors(index) -> int:
    index_stats = index.describe_index_stats()
    num_vectors = index_stats['total_vector_count']
    return num_vectors


def delete_all_vectors(index) -> None:
    index.delete(delete_all=True)
