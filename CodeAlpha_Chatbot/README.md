# CodeAlpha Chatbot for FAQs

This project is a command-line interface (CLI) chatbot designed to answer Frequently Asked Questions (FAQs) about Blockchain technology. It was built as part of the CodeAlpha Artificial Intelligence Internship.

## 🌟 Features
- **NLP Preprocessing:** Uses the Natural Language Toolkit (NLTK) to tokenize, remove stop words, and lemmatize user input and FAQ data.
- **Intent Matching:** Utilizes Term Frequency-Inverse Document Frequency (TF-IDF) and Cosine Similarity via `scikit-learn` to find the most mathematically similar question in the dataset.
- **Interactive CLI:** A simple and continuous command-line loop allowing users to ask multiple questions.

## 🛠️ Technologies Used
- Python 3.x
- `nltk` (Natural Language Toolkit)
- `scikit-learn` (for TF-IDF and Cosine Similarity)
- JSON (for FAQ data storage)

## 🚀 How to Run

1. **Install Dependencies:**
   Ensure you have the required libraries installed:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The script will automatically download the necessary NLTK datasets on its first run).*

2. **Run the Chatbot:**
   ```bash
   python main.py
   ```

3. **Interact:**
   Type your questions about blockchain into the terminal. Type `quit` to exit the application.
