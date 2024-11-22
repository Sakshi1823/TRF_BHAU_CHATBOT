import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
import os
import pandas as pd
import warnings

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import SQLChatMessageHistory
from pydantic import BaseModel

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables 
load_dotenv('.env')
api_key = os.getenv('API_KEY')
app = FastAPI()

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request
class QueryRequest(BaseModel):
    query: str
    session_id: str

model = ChatGoogleGenerativeAI(model="gemini-pro", 
                              google_api_key=api_key,
                              temperature=0.5, 
                              convert_system_message_to_human=True)

# Load and process workshop data
with open('workshop_data.txt', 'r') as file:
    workshop_data = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=3)
texts = text_splitter.split_text(workshop_data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                                        google_api_key=api_key)
vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 3})

# Modified template to include chat history
template = """
    Previous conversation history: {chat_history}
    
    a. You are an intelligent agent with two characters:
    [Rest of your existing template...]
    
    Remember the context of our previous conversation while answering.
    
    {context}
    Question: {question}
    Helpful Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=template
)

# Function to get memory for a session
def get_memory(session_id: str):
    return ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=SQLChatMessageHistory(
            session_id=session_id,
            connection_string="sqlite:///chat_history.db"
        ),
        return_messages=True
    )

@app.post("/bhau_api")
async def respond_to_doubts(query_request: QueryRequest):
    # Get memory for this session
    memory = get_memory(query_request.session_id)
    
    # Create QA chain with memory
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": QA_CHAIN_PROMPT,
            "memory": memory,
        }
    )

    # Get response
    result = qa_chain({
        "query": query_request.query,
    })

    # Save the interaction to memory
    memory.save_context(
        {"input": query_request.query},
        {"output": result['result']}
    )

    return {
        "result": result['result'],
        "session_id": query_request.session_id
    }

# Endpoint to clear conversation history
@app.post("/clear_history/{session_id}")
async def clear_history(session_id: str):
    memory = get_memory(session_id)
    memory.clear()
    return {"message": f"Conversation history cleared for session {session_id}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)