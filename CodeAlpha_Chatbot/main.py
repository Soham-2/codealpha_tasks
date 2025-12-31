import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

with open('faq_data.json', 'r', encoding='utf-8') as f:
    faqs = json.load(f)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

preprocessed_questions = [preprocess_text(faq['question']) for faq in faqs]

tfidf_vectorizer = TfidfVectorizer()
faq_vectors = tfidf_vectorizer.fit_transform(preprocessed_questions)

def get_best_match(user_question, threshold=0.3):
    user_question_preprocessed = preprocess_text(user_question)
    user_question_vector = tfidf_vectorizer.transform([user_question_preprocessed])

    similarities = cosine_similarity(user_question_vector, faq_vectors)
    max_similarity = similarities.max()

    if max_similarity >= threshold:
        best_match_index = similarities.argmax()
        return faqs[best_match_index]['answer']
    else:
        return "I'm sorry, I don't have an answer for that. Please try rephrasing your question."

print("Welcome to the Blockchain FAQ Chatbot! Ask me anything about Blockchain. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Chatbot: Goodbye!")
        break
    response = get_best_match(user_input)
    print(f"Chatbot: {response}")
