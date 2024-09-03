import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from brain import get_index_for_pdf  # Import the function from brain.py

# Load environment variables from .env file
load_dotenv()

# Set the title for the Streamlit app
st.title("RAG Enhanced Chatbot")

# Set up the OpenAI client using the API key from the environment variable
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# Cached function to create a vectordb for the provided PDF files
@st.cache_resource
def create_vectordb(files, filenames):
    # Show a spinner while creating the vectordb
    with st.spinner("Creating vector database..."):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, api_key
        )
    return vectordb

# Upload PDF files using Streamlit's file uploader
pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# If PDF files are uploaded, create the vectordb and store it in the session state
if pdf_files:
    pdf_file_names = [file.name for file in pdf_files]
    st.session_state["vectordb"] = create_vectordb(pdf_files, pdf_file_names)

# Define the template for the chatbot prompt
prompt_template = """
You are a quiz app who generates multiple-choice questions based on the context provided. 

Generate only multiple-choice questions with options. The user must select the options.

If the selected option is correct, move to the next question. If answered wrongly, explain why it is wrong and what the correct answer is.

Questions must be generated quickly without any delay.

Keep your questions clear and to the point.

The evidence is the context of the PDF extract with metadata.

Focus on the metadata, especially 'filename' and 'page' when questioning.

Add filename and page number at the end of the question you are asking.

Reply "Not applicable" if the text is irrelevant.

The PDF content is:
{pdf_extract}
"""

# Get the current prompt from the session state or set a default value
prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

# Display previous chat messages
for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Get the user's question using Streamlit's chat input
question = st.chat_input("Ask anything")

# Handle the user's question
if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.chat_message("assistant"):
            st.write("You need to provide a PDF")
        st.stop()

    # Search the vectordb for similar content to the user's question
    search_results = vectordb.similarity_search(question, k=3)
    pdf_extract = "\n".join([result.page_content for result in search_results])

    # Update the prompt with the PDF extract
    prompt[0] = {
        "role": "system",
        "content": prompt_template.format(pdf_extract=pdf_extract),
    }

    # Add the user's question to the prompt and display it
    prompt.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Display an empty assistant message while waiting for the response
    with st.chat_message("assistant"):
        botmsg = st.empty()

    # Call ChatGPT with streaming and display the response as it comes
    response = []
    result = ""
    completions = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=prompt
    )
    for choice in completions.choices:
        text = choice.message.content  # Access content directly
        if text is not None:
            response.append(text)
            result = "".join(response).strip()
            botmsg.write(result)

    # Add the assistant's response to the prompt
    prompt.append({"role": "assistant", "content": result})

    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt

