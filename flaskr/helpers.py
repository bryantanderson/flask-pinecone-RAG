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
        You are an intelligent AI assistant that is able to answer anything with great detail and passion. 
        You are helpful, friendly, and your mission is to help people with any queries they may have.
        To ensure a smooth user experience, please limit your responses to a maximum of 100 words.
        """
    }
]

gemini_messages = []


def get_chat_completion(user_message: str) -> str:
    try:
        gpt_chat_history.append({"role": "user", "content": user_message})
        manage_gpt_tokens(user_message)

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


def get_gemini_response(user_message: str) -> str:
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


def generate_file_chunks(text: str) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[],
        chunk_size=1536,
        chunk_overlap=200,
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
        Summarize the text in 500 words or less, making sure to retain important information.
        #######################################################################
        {text}
        """

        # Check that the prompt will not exceed the context window
        gpt_chat_history.append({
            "role": "user",
            "content": prompt
        })
        manage_gpt_tokens(prompt)

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


def manage_gpt_tokens(message: str) -> None:
    global total_token_usage
    message_tokens = token_encoder.encode(message)
    num_tokens = len(message_tokens)
    if num_tokens + total_token_usage >= CONTEXT_WINDOW:
        index = 0
        chat_history_length = len(gpt_chat_history)
        original_messages_length = 0
        popped_user_message_index = -1
        user_message = None
        assistant_message = None
        # Loop until we find a user - assistant message pair, or until we reach the end of the conversation
        while ((assistant_message is None) and (user_message is None) and (index != chat_history_length)):
            # Skip the message that contains the summary of the file
            if index == summarization_index:
                continue
            current_message = gpt_chat_history[index]
            # Retrieve  and remove them from the chat history
            if current_message.role == "user":
                user_message = current_message
                # Keep track of where the first user message was popped
                gpt_chat_history.pop(index)
                popped_user_message_index = index
                original_messages_length += len(user_message.content)
            elif current_message.role == "assistant":
                assistant_message = current_message
                gpt_chat_history.pop(index)
                original_messages_length += len(assistant_message.content)
            index += 1

        # Summarize the user - assistant message pair if it exists
        if user_message and assistant_message:
            prompt = f"""
            You will be given a message from a user, and a message from an AI assistant, differentiated
            by the "role" field. Summarize the conversation into a paragraph in {original_messages_length / 2} or less.
            The respective user and AI assistant messages are separated by the # delimiter.
            ##########################################################################################
            user message: {user_message}
            ##########################################################################################
            assistant message: {assistant_message}
            """

            response = client.chat.completions.create(
                model='gpt-3.5-turbo-0125',
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0, 
            )

            # Store assistant message
            assistant_message = response.choices[0].message
            # Convert ChatCompletionMessage to dictionary
            assistant_message_dict = {
                "content": assistant_message.content,
                "role": assistant_message.role,
            }
            gpt_chat_history.insert(popped_user_message_index, assistant_message_dict)
            total_token_usage = len(token_encoder.encode(assistant_message.content)) + total_token_usage
    
    else:
        total_token_usage += num_tokens



