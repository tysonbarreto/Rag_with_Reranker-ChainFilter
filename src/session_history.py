from langchain_community.chat_message_histories import ChatMessageHistory

store ={}

def get_session_history(session_id:str)->ChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

if __name__=="__main__":
    __all__=["get_session_history"]