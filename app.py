import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith tracking setup
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OpenAI"

## Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant please reponse to the user queries."),
        ("user", "Question: {question}"),
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

# Streamlit app

## title 
st.title("Q&A Chatbot with OpenAI")

## Sidebar for settings
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

### select LLM model
llm = st.sidebar.selectbox("Select LLM Model", ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro"])

### adjust response parameters
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.sidebar.slider("Max Tokens", 50, 1000, 200, 50)

## Main interface for user input
st.write("Go ahead and ask any question!")
user_input = st.text_input("You:")

if user_input and api_key:
    with st.spinner("Generating response..."):
        response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.text_area("Bot:", value=response)
else:
    st.info("Please enter your OpenAI API Key and ask a question.")

