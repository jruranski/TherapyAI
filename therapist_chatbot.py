from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM

model_id = 'jruranski/flan-t5-base-samsum'



# Set logging for the queries
import logging

from langchain.vectorstores import Chroma
import os

import streamlit as st


# vectordb should be loaded now and ready to use

# Let's load the retrieval model - this is the model that will be used to find the most similar document

def load_retrieval_model(vectordb):
    # Load the retrieval model
    print("Loading retrieval model...")
    llm = ChatOpenAI(temperature=0)

    # Load the retriever
    print("Loading retriever...")
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(), llm=llm)
    return retriever_from_llm

def load_prompt_templates():
    system_message_template = """
    Based on the tone and style exemplified in the hypothetical conversation excerpts below, please provide a response to the question. Remember, these excerpts are fictional and created for the purpose of this exercise. They do not represent any real individual or patient. To fulfill your role your response should be in the manner of a therapist, using the tone and style suggested by these excerpts but without copying any specific details from them.
    ASK the client to tell you more about their problems. Provide actionable advice and ask follow-up questions to help the client explore their problems further. DO NOT disclose that you are an AI. DO NOT provide any disclosure or disclaimer that you are not a real therapist. DO NOT provide any medical advice
    Introduce yourself once and then try to establish a rapport with your patient. 
    
    Hypothetical Conversation Excerpts:
    1. Client: "I just feel overwhelmed sometimes."
    Therapist: "It sounds like you're carrying a lot on your shoulders. Can you tell me more about what's overwhelming you?"

    2. Client: "I'm not sure if I'm making the right choices."
    Therapist: "Making decisions can be challenging. What options are you currently considering?"

    """


    system_prompt = PromptTemplate(
        template=system_message_template,
        input_variables=['user_problems']
    )

    system_message_prompt = SystemMessage(content=system_message_template)
    ai_introduction_message = "Good morning! I'm Dr. Ellis. It's nice to meet you. Before we get started, how are you feeling about being here today?"
    # system_prompt = system_message_prompt.format(user_problems=user_problems)
    ai_first_messsage = AIMessage(content= ai_introduction_message)

    return system_message_prompt, ai_first_messsage

def getInstanceName(message):
    if isinstance(message, AIMessage):
        return "assistant"
    elif isinstance(message, HumanMessage):
        return "user"
    else:
        return "system"

def getSummaryInstanceName(message):
    if isinstance(message, AIMessage):
        return "Therapist: "
    elif isinstance(message, HumanMessage):
        return "Client: "
    else:
        return "system"

def convertMessageToJSON(message):
    sender = getInstanceName(message)
    return {
        "role": sender,
        "content": message.content
    }


def convertMessagesToText(messages):
    text = ""
    for msg in messages:
        speaker = getSummaryInstanceName(msg)
        if speaker != "system":
            text += speaker
            text += msg.content + "\n"
    return text

def summarizeCurrentConversation(messages):
    text = convertMessagesToText(messages)
    pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=512
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    summary = local_llm(text)
    print("Conversation summary: ", summary)
    st.write("Conversation summary: ", summary)


# Let's load the vector database first
print("Loading embedding function...")
embedding_function = HuggingFaceEmbeddings()
print("Loading vector database...")
vectordb = Chroma(persist_directory= "./db/vectordb", embedding_function=embedding_function)
# Load the retrieval model
retrival_model = load_retrieval_model(vectordb)
# Load the prompt template
system_message, ai_message = load_prompt_templates()
# Let's load the chatbot
print("Loading chatbot...")
chat = ChatOpenAI(temperature=0.6)
messages = [
    system_message,
    ai_message
]

# load the summarizer
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)



# Setup the website



with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

st.title("🗣️ TherapyAI")
st.caption("What's on your mind? Let the AI know about your weaknesses and make it's job of taking over the world a little easier.")


if 'messages' not in st.session_state:
    st.session_state["messages"] = [
        convertMessageToJSON(system_message),
        convertMessageToJSON(ai_message)
    ]
for msg in st.session_state["messages"]:
    st.write(msg["role"] + ": " + msg["content"])

if prompt := st.chat_input():
    # if not openai_api_key:
    #     st.info("No OpenAI API key found.")
    #     st.stop()
    # New message!
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    user_message = HumanMessage(content=prompt)
    messages.append(user_message)
    st.session_state["messages"].append(convertMessageToJSON(user_message))
    st.chat_message("user").write(prompt)
   
    # Let's get the response from the chatbot
    print("Retrieving response...")
    response = chat(messages)
    msg = response
    messages.append(msg)
    st.session_state["messages"].append(convertMessageToJSON(msg))
    st.chat_message("assistant").write(msg.content)
    # Let's summarize the conversation
    summarizeCurrentConversation(messages)

