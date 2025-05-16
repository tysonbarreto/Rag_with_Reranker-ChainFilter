from typing import Optional

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_filter import LLMChainFilter
from langchain_core.language_models import BaseLanguageModel
from langchain.vectorstores.base import VectorStore, VectorStoreRetriever
from langchain_qdrant import Qdrant

from src.config import Config
from src.model import create_embeddings, create_llm, create_reranker

def create_retriever(llm:BaseLanguageModel, vector_store:Optional[VectorStore]=None)->VectorStoreRetriever:
    if not vector_store:
        vector_store = Qdrant.from_existing_collection(
            embedding=create_embeddings(),
            path=Config.Path.DOCUMENTS_DIR,
            collection_name=Config.Database.DOCUMENTS_COLLECTION
        )
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":5})

    if Config.Retriever.USE_RERANKER:
        retriever = ContextualCompressionRetriever(base_compressor=create_reranker(), base_retriever=retriever)

    if Config.Retriever.USE_CHAIN_FILTER:
        retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainFilter.from_llm(llm=llm), base_retriever=retriever
        )
    
    return retriever

if __name__=="__main__":
    __all__=["create_retriever"]