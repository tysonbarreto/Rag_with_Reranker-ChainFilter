import re
from operator import itemgetter
from typing import List
from textwrap import dedent

from langchain.schema.runnable import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import Config
from src.session_history import get_session_history

SYSTEM_PROMPT=dedent("""
                        Utilize the provided contextual information to respond to the user question.
                        If the answer is not found within the context, state that the answer cannot be found.
                        Prioritize concise responses (maximum of 3-5 sentences) and use a list where applicable.
                        The contextual information is organized with the most relevant source appearing first.
                        Each source is separated by a horizontal rule (---).
                        
                        Context:
                        {context}

                        Use markdown formatting where appropriate.
                    """)

def remove_links(text:str)->str:
    url_pattern = r"https?:\/\/\S+|www\.\S+"
    return re.sub(url_pattern,"",text)

def format_documents(documents:List[Document])->str:
    texts=[]
    for doc in documents:
        texts.append(doc.page_content)
        texts.append("---")
    
    return remove_links("\n".join(texts))

def create_chain(llm:BaseLanguageModel, retriever:VectorStoreRetriever)->Runnable:

    prompt=ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question")
            | retriever.with_config({"run_name":"context_retriever"})
            | format_documents
        )
        | prompt
        | llm
    )

    return RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    ).with_config({"run_name":"chain_answer"})


async def ask_question(chain:Runnable, question:str, session_id:str):
    async for event in chain.astream_events(input={"question":question},
                                             config={
                                                 "callbacks":[ConsoleCallbackHandler()] if Config.DEBUG else [],
                                                 "configurable": {"session_id":session_id}
                                             },
                                             version="v2",
                                             include_names=["context_retreiver", "chain_answer"]):
        event_type=event["event"]
        if event_type=="on_retriever_end":
            yield event["data"]["output"]
        if event_type=="on_chain_stream":
            yield event["data"]["chunk"].content

if __name__=="__main__":
    __all__=["SYSTEM_PROMPT","create_chain","format_documents","remove_links","ask_question"]