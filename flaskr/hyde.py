import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder

load_dotenv()

llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
base_embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

def get_hypothetical_response_embedding(user_query):
    """
    Enhance retrieval by generating a hypothetical document for an incoming query.
    The process initiates with a Large Language Model (LLM), such as ChatGPT, tasked with 
    crafting a document based on a specific question or subject. While this artificially created 
    document might contain inaccuracies, it encapsulates patterns and nuances that resonate 
    with similar documents in a reliable knowledge base.
    """

    prompt_template = """Please answer the user's question about the details regarding important work documents
    Question: {question}
    Answer:"""

    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain, 
        base_embeddings=base_embeddings
    )
    result = embeddings.embed_query(user_query)
    return result
