# Mission Quizify

Welcome to the **Mission Quizify** project! This repository contains the implementation of a quiz generation and PDF chat application using Streamlit and Langchain, integrated with Google Cloud's VertexAI. The project is divided into several tasks, each building upon the previous one to create a robust system for quiz generation and interactive PDF content querying.

## Table of Contents

- [Introduction](#introduction)
- [Tasks Overview](#tasks-overview)
  - [Task 3: PDF Processing](#task-3-pdf-processing)
  - [Task 4: Embedding Client](#task-4-embedding-client)
  - [Task 5: Chroma Collection Creator](#task-5-chroma-collection-creator)
  - [Task 6: Quiz Builder](#task-6-quiz-builder)
  - [Task 7: Quiz Generator](#task-7-quiz-generator)
  - [Task 8: Quiz Generation and Validation](#task-8-quiz-generation-and-validation)
  - [Task 9: Quiz Manager](#task-9-quiz-manager)
  - [Task 10: Complete Quiz Application](#task-10-complete-quiz-application)
  - [Task 11: Chat with PDF](#task-11-chat-with-pdf)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Mission Quizify is an interactive application designed to allow users to upload PDF documents, generate quizzes based on the content, and interactively chat with the PDF content. This project leverages Streamlit for the web interface and integrates Langchain and Google Cloud's VertexAI for advanced NLP capabilities.

## Tasks Overview

### Task 3: PDF Processing

**File:** `tasks/task_3/task_3.py`

This task involves creating a `DocumentProcessor` class that processes uploaded PDF documents using Langchain's `PyPDFLoader`. It extracts the pages from the PDFs and displays the total number of pages processed.

### Task 4: Embedding Client

**File:** `tasks/task_4/task_4.py`

The `EmbeddingClient` class is initialized to connect to Google Cloud's VertexAI for text embeddings. It handles embedding queries and documents using the specified model.

### Task 5: Chroma Collection Creator

**File:** `tasks/task_5/task_5.py`

The `ChromaCollectionCreator` class creates a Chroma collection from the documents processed by the `DocumentProcessor`. It splits the documents into text chunks and indexes them using the embedding model.

### Task 6: Quiz Builder

**File:** `tasks/task_6/task_6.py`

This task involves building a `Quiz Builder` application with Streamlit. It uses the `DocumentProcessor`, `EmbeddingClient`, and `ChromaCollectionCreator` to allow users to upload documents, specify a quiz topic, and generate quizzes.

### Task 7: Quiz Generator

**File:** `tasks/task_7/task_7.py`

The `QuizGenerator` class is designed to generate quiz questions based on the topic provided. It integrates with the vectorstore to retrieve relevant context and uses the LLM to generate structured quiz questions in JSON format.

### Task 8: Quiz Generation and Validation

**File:** `tasks/task_8/task_8.py`

This task extends the `QuizGenerator` class to validate the uniqueness of generated quiz questions and compile a list of unique questions for the quiz. It includes methods for question generation and validation.

### Task 9: Quiz Manager

**File:** `tasks/task_9/task_9.py`

The `QuizManager` class is responsible for managing the quiz questions. It includes methods to navigate through the quiz, retrieve specific questions by index, and handle quiz operations.

### Task 10: Complete Quiz Application

**File:** `tasks/task_10/task_10.py`

This task integrates all previous tasks to create a complete quiz application. It includes the entire workflow from PDF uploading, processing, quiz generation, and quiz management using Streamlit.

### Task 11: Chat with PDF

**File:** `tasks/task_11/task_11.py`

The final task adds a `ChatWithPDF` feature that allows users to interactively ask questions about the PDF content. It integrates the previously created components and uses the LLM to provide concise answers based on the PDF content.

## Setup and Installation

To run this project, follow these steps:

1. **Clone the repository:**

   ```sh
   git clone https://github.com/your-username/mission-quizify.git
   cd mission-quizify
  
2. **Create and activate a virtual environment:**
   ```
   python3 -m venv venv
   source venv/bin/activate

3. **Install the required dependencies:**

   ```
   pip install -r requirements.txt

4. **Run the Streamlit application:**

  ```
streamlit run tasks/task_11/task_11.py
```

## Usage

1. **Upload PDF Documents:**
   Use the interface to upload PDF documents and process them.

2. **Generate Quizzes:**
   Specify the quiz topic and number of questions, then generate the quiz.

3. **Chat with PDF:**
   Use the chat interface to ask questions about the PDF content and receive answers.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or additions.

## License:

This project is licensed under the MIT License.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

This project, Gemini Quizify, was completed as part of the mission in the [Radical AI app](https://ai.radicalai.app/Gemini-Quizify-2024-Feb-28).







   

