import os
import streamlit as st
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from streamlit.logger import get_logger
from chains import load_embedding_model, load_llm
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

# Load embedding model
embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)
print(f"embedding is done {embeddings}")

class StreamHandler(BaseCallbackHandler):
    """Custom stream handler for Streamlit."""
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Append new token to text and update container."""
        self.text += token
        self.container.markdown(self.text)


# Load LLM
llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})


def main():
    st.header("📄 Chat with Your PDF File")

    # Upload your PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = "".join([page.extract_text() for page in pdf_reader.pages])

        # LangChain text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        print(f"the chunks are {chunks}")
        # Store chunks in Neo4j VectorStore
        vectorstore = Neo4jVector.from_texts(
            chunks,
            url=url,
            username=username,
            password=password,
            embedding=embeddings,
            index_name="pdf_bot",
            node_label="PdfBotChunk",
            pre_delete_collection=True,  # Clear existing data
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )
        print(f"the qa is {qa}")
        # Accept user query
        query = st.text_input("Ask questions about your PDF file")
        
        if query:
            print(f"the query is {query}")
            docs = vectorstore.as_retriever().get_relevant_documents(query)
            print(docs)
            # Stream handler for real-time response updates
            stream_handler = StreamHandler(st.empty())
            with st.spinner("Processing your query..."):
                response = qa.run(query, callbacks=[stream_handler])
                print("response is {response}")
            if response:
                st.success("Answer retrieved!")
            else:
                st.warning("No response generated. Try refining your query.")
        else:
            print(f"there is no query")

if __name__ == "__main__":
    main()
