import streamlit as st
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator
from tasks.task_8.task_8 import QuizGenerator


class QuizManager:
    def __init__(self, questions: list):
        """
        Initializes the QuizManager class with a list of quiz questions.

        Parameters:
        - questions: A list of dictionaries, where each dictionary represents a quiz question along with its choices, correct answer, and an explanation.
        """
        self.questions = questions
        self.total_questions = len(questions)

    def get_question_at_index(self, index: int):
        """
        Retrieves the quiz question object at the specified index. If the index is out of bounds,
        it restarts from the beginning index.

        :param index: The index of the question to retrieve.
        :return: The quiz question object at the specified index, with indexing wrapping around if out of bounds.
        """
        valid_index = index % self.total_questions
        return self.questions[valid_index]

    def next_question_index(self, direction=1):
        """
        Adjust the current quiz question index based on the specified direction.

        :param direction: An integer indicating the direction to move in the quiz questions list (1 for next, -1 for previous).
        """
        current_index = st.session_state.get("question_index", 0)
        new_index = (current_index + direction) % self.total_questions
        st.session_state["question_index"] = new_index


if __name__ == "__main__":

    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "gemini-quizify-427418",
        "location": "us-central1"
    }

    screen = st.empty()
    with screen.container():
        st.header("Quiz Builder")
        processor = DocumentProcessor()
        processor.ingest_documents()

        embed_client = EmbeddingClient(**embed_config)

        chroma_creator = ChromaCollectionCreator(processor, embed_client)

        question = None
        question_bank = None

        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")

            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)

            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()

                st.write(topic_input)

                # Test the Quiz Generator
                generator = QuizGenerator(topic_input, questions, chroma_creator)
                question_bank = generator.generate_quiz()

    if question_bank:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Question: ")

            quiz_manager = QuizManager(question_bank)

            with st.form("Multiple Choice Question"):
                index_question = quiz_manager.get_question_at_index(st.session_state.get("question_index", 0))

                choices = []
                for choice in index_question['choices']:
                    key = choice["key"]
                    value = choice["value"]
                    choices.append(f"{key}) {value}")

                st.write(index_question["question"])

                answer = st.radio(
                    'Choose the correct answer',
                    choices
                )
                submit_answer = st.form_submit_button("Submit Answer")

                if submit_answer:
                    correct_answer_key = index_question['answer']
                    if answer.startswith(correct_answer_key):
                        st.success("Correct!")
                    else:
                        st.error("Incorrect!")
                    st.session_state["show_explanation"] = True

                if st.session_state.get("show_explanation", False):
                    st.write(f"Explanation: {index_question['explanation']}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Previous Question"):
                    quiz_manager.next_question_index(-1)
                    st.session_state["show_explanation"] = False
                    st.experimental_rerun()

            with col2:
                if st.button("Next Question"):
                    quiz_manager.next_question_index(1)
                    st.session_state["show_explanation"] = False
                    st.experimental_rerun()
