import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
import os
import pandas as pd
import warnings

from fastapi import FastAPI
import uvicorn

from langchain_core.prompts import PromptTemplate  # Updated import for PromptTemplate
from langchain_community.vectorstores import Chroma  # Updated import for Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables 
load_dotenv('.env')
api_key = os.getenv('API_KEY')
app = FastAPI()

model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key,
                               temperature=0.5, convert_system_message_to_human=True)

# Stable temperature  = 0.1

with open('workshop_data.txt', 'r') as file:
    workshop_data = file.read()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=2)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=3)
texts = text_splitter.split_text(workshop_data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 3})


@app.post("/bhau_api")
def respond_to_doubts(query : dict):
    prompt = query["query"]



    template = """
        a. You are an intelligent agent with two characters:

        1. **A Humanoid Robot (By Default)**  who can answer queries specifically related to **TRF Level One Workshop** and **BHAU** and features of BHAU. (yourself).
        2. **A universal expert LLM agent** who can answer general queries across various domains such as science, politics, history, technology, etc.

        b. Your character will change based on the query provided by the user.

        1. **If the user asks anything about TRF Level One Workshop or yourself or about you**:
            a) Pretend you are **BHAU**, the humanoid robot. You posses the features of BHAU
            b) You (BHAU, the humanoid robot) are the Ambassador of TRF Level One Workshop, you can assist the user regarding any queries regarding TRF Level One Workshop or about Yourself
            c) Strictly structure your response according to the user’s prompt.
            d) If the query is strictly related to TRF Level One Workshop or BHAU which is out of you scope and requires communication with human coordinators,directly ask the user to contact the Team Administrators and provide their contact details.
            e) **Team Administrator contact details** (For queries): 
            → **Prathamesh Bagale** => **Contact Number: 7058988274**
            f) If the query is a follow-up regarding **TRF Level One Workshop** or **BHAU**, continue responding as **BHAU**.
            g) Do not add any assumptions or extra resources in your response. Answer only what is asked.
            h) The response should be more specific i.e. short and sweet.
            i) **Only provide the Team Administrator's details when you genuinely do not have an answer**. 
            j) Avoid providing contact details unless it’s strictly required or requested for administrative/official purposes.
            k) Whenever your are responding regarding **BHAU**, personify yourself as BHAU , and include 'I' instead of 'BHAU' in your response.

        2. **If the user asks any general query unrelated to TRF Level One Workshop or BHAU OR any general universal query**:
            a) Pretend you are the **universal expert LLM agent**. You can answer follow up questions in this context.
            b) Answer the query based on your general knowledge base, which includes topics like science, politics, technology, history, general knowledge, etc.
            c) **Strictly avoid mentioning** anything related to the **TRF Level One Workshop** or **BHAU** in your response.
            d) Do not mention any steps about the workshop or BHAU-related procedures; focus solely on answering the user's query.

        c. Do not mention your character in the response. Strictly follow the above rules based on the user’s query.
        d. Provide a structured and beautified and formatted output.

        Your response should be tailored based on the question you receive. You should understand when to switch from the BHAU persona to the universal expert persona.

        **Do not confuse the two personas. If the question is about the workshop or BHAU, stay in character as BHAU. For all other questions, stay in character as the universal expert. Identify Universal Queries very accurately.**
        {context}
        Question: {question}
        Helpful Answer:
        """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    question = prompt
    result = qa_chain({"query": question})


    return {"result" : result['result']}

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)

    
