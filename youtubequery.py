subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
from langchain.document_loaders import YoutubeLoader
from langchain import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import streamlit as st
import os



api_key = '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'

# Display the Page Title
st.title(' Youtube QnA')

llm = OpenAI(temperature=0)  # Temp controls the randomness of the text

prompt = st.text_input("Paste the URL")

if prompt:
    loader = YoutubeLoader.from_youtube_url(prompt, add_video_info=False)
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    split_docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key= api_key)
    doc_search = Chroma.from_documents(docs,embeddings)
    chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=doc_search.as_retriever())

    query = st.text_input("Enter a query:")

    if st.button("Execute"):
        answer = chain.run(query)
        st.write("Answer:")
        st.write(answer)
