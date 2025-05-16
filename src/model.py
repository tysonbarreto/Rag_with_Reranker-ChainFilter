from langchain_community.chat_models import AzureChatOpenAI, ChatOllama
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import AzureChatOpenAI
# from langchain_groq import ChatGroq
from groq import Groq
import os

from src.config import Config


def create_llm()->BaseLanguageModel:
    return Groq(api_key=os.environ.get("GROQ_API_KEY"),base_url=Config.Model.REMOTE_LLM)

def create_embeddings()-> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)

def create_reranker()->FlashrankRerank:
    return FlashrankRerank(model=Config.Model.RERANKER)

if __name__=="__main__":
    __all__=["create_llm","create_embeddings","create_reranker"]