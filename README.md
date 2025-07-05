# OpenCampus-Chatbot
------------------------

Welcome to OpenCampus Bot! This is an intelligent chatbot designed to help you quickly find information about courses offered by OpenCampus. Whether you want to know what courses are available, get detailed information about a specific program, or compare two courses side-by-side, OpenCampus Bot is here to assist.

---

<img width="1029" alt="Screenshot 2025-07-05 at 22 37 17" src="https://github.com/user-attachments/assets/db72bc81-0209-4343-99da-06a3774a957f" />

-----------

## What is Opencampus Bot?

Opencampus Bot acts as your personal guide to OpenCampus courses. Instead of manually searching through the website, you can simply ask OpenCampus Bot your questions in plain language, and it will provide relevant answers based on the official course materials.

---

## Features

* **Course Listings:** Ask for a general list of courses or courses within specific categories (e.g., "What courses do you offer?", "Show me courses in Tech & Coding?").
* **Detailed Course Information:** Get comprehensive details about any specific course (e.g., "Tell me about Python Programming?", "What's the schedule for Machine Learning?").
* **Course Comparison:** Compare two courses side-by-side to see their descriptions, key learnings, ECTS credits, and more (e.g., "Compare R and Python?", "How is Web Development different from Data Science?").
* **Intelligent Understanding:** Uses advanced AI to understand your questions and find the most relevant information.

---

## How It Works (Behind the Scenes)

Opencampus Bot works in three main steps:

### 1. Gathering Course Information (The "Collector")

* **What it does:** We have a special script that acts like a web browser, visiting the OpenCampus website. It carefully reads through all the course pages, collecting every detail about each course – like its title, what you'll learn, who teaches it, where it's held, and even the session dates.
* **Why it's important:** This step ensures Opencampus Bot has all the latest and most accurate information directly from the source.
* **Output:** All this collected information is neatly organized and saved into structured `.json` files on your computer.

### 2. Building the Brain (The "Librarian")

* **What it does:** Another script takes all those collected `.json` files. It reads the course details and converts them into a special numerical format that computers can understand very quickly. Think of it like creating a super-fast, searchable index for a massive library. This index is called a "vector database" (specifically, a FAISS index).
* **Why it's important:** When you ask a question, EduBot doesn't read every single course description from scratch. Instead, it uses this "brain" to instantly find only the most relevant pieces of information related to your question.
* **Output:** A ready-to-use "brain" (the `faiss_index` folder) that the chatbot can query.

### 3. Talking to the Bot (The "Chat Interface")

* **What it does:** This is the part you interact with! When you type a question:
    * **Understanding Your Intent:** A smart AI part first figures out *what you want to do*. Are you asking for a simple list? Detailed info? Or do you want to compare courses?
    * **Finding the Answer:**
        * If it's a general question or detail request, it uses the "brain" (FAISS index) to pull out the most relevant course information.
        * If you're comparing courses, it has a special process to find the exact courses you mean (even if you use short names) and then creates a neat comparison table for you.
    * **Giving You the Response:** Finally, it uses powerful AI language models (like Google Gemini or local models like Gemma/Llama) to turn the found information into a clear, natural-sounding answer that's displayed in the chat.
* **Why it's important:** This is where all the pieces come together to provide a seamless, intelligent conversation experience.

---

## Technologies Used

* **Python:** The main programming language.
* **Playwright:** For web scraping (simulating a browser).
* **LangChain:** A framework for building LLM applications.
* **FAISS:** For fast similarity search in the knowledge base.
* **Google Generative AI Embeddings:** To convert text into searchable numerical data.
* **Google Gemini / Ollama:** The powerful AI models that understand and generate text.
* **Streamlit:** For creating the interactive web application (the chat interface).

---

## How to Run Opencampus Bot (Quick Start)

To get EduBot up and running, you'll need Python installed and a few simple steps:

1.  **Get the Code:** Download or clone this project to your computer.
2.  **Install Dependencies:** Open your terminal or command prompt in the project folder and run:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set Your Google API Key:** Open `ingest_data.py` and `main.py`. Find the line `GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"` and replace `"YOUR_GOOGLE_API_KEY"` with your actual Google API Key.
4.  **Gather Course Data:** Run the scraping script (this will create the `domain_data` folder and fill it with course info):
    ```bash
    python scraping_pipeline.py
    ```
5.  **Build the Knowledge Base:** Run the ingestion script (this will create the `faiss_index` folder):
    ```bash
    python ingest_data.py
    ```
6.  **Start the Chatbot:** Finally, launch the Streamlit application:
    ```bash
    streamlit run main.py
    ```
    Your opencampus chatbot will open in your web browser!

**Note:** If you want to use local AI models (like Gemma or Llama), you'll also need to set up [Ollama](https://ollama.com/) and pull the models you want to use before running `main.py`.

---

Enjoy chatting with OpenCampus Bot!

