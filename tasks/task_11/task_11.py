import streamlit as st
import os
import sys
import json
from langchain_google_vertexai import VertexAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator
from tasks.task_8.task_8 import QuizGenerator

class ChatWithPDF:
    def __init__(self, processor, embed_client, chroma_creator):
        self.processor = processor
        self.embed_client = embed_client
        self.chroma_creator = chroma_creator
        self.llm = VertexAI(
            model_name="gemini-pro",
            temperature=0.5,
            max_output_tokens=500
        )

    def generate_response(self, user_input):
        if not self.chroma_creator.db:
            st.error("Chroma collection is not created.")
            return "Chroma collection is not available."

        retriever = self.chroma_creator.db.as_retriever()
        documents = retriever.get_relevant_documents(user_input)
        context = " ".join([doc.page_content for doc in documents])

        prompt_template = """
        You are an assistant that can answer questions based on the content of a PDF document.
        Provide a helpful and concise answer to the user's query.

        Context: {context}
        Query: {query}
        """
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm

        response = chain.invoke({"context": context, "query": user_input})
        return response

if __name__ == "__main__":
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "gemini-quizify-427418",
        "location": "us-central1"
    }

    st.header("Quizzify - Chat with PDF")

    screen = st.empty()
    with screen.container():
        st.header("Upload and Process PDF")
        processor = DocumentProcessor()
        processor.ingest_documents()

        embed_client = EmbeddingClient(**embed_config)
        chroma_creator = ChromaCollectionCreator(processor, embed_client)
        chroma_creator.create_chroma_collection()

        with st.form("Load Data to Chroma"):
            st.subheader("Upload PDFs and Create Chroma Collection")
            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.write(topic_input)
                generator = QuizGenerator(topic_input, questions, chroma_creator)
                question_bank = generator.generate_quiz()

    st.header("Chat with PDF")
    chat = ChatWithPDF(processor, embed_client, chroma_creator)

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    with st.container():
        user_input = st.text_input("Ask a question about the PDF:")
        if st.button("Send"):
            if user_input:
                st.session_state['messages'].append({"role": "user", "content": user_input})
                response = chat.generate_response(user_input)
                st.session_state['messages'].append({"role": "assistant", "content": response})

        for message in st.session_state['messages']:
            if message['role'] == 'user':
                with st.container():

                    st.write(f"**You:** {message['content']}")
            else:
                with st.container():

                    st.write(f"**Assistant:** {message['content']}")
