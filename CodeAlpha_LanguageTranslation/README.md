# CodeAlpha Language Translation Tool

This project is a web-based Language Translation Tool built using Streamlit and the Google Translate API. It was created as part of the CodeAlpha Artificial Intelligence Internship.

## 🌟 Features
- **Interactive Web UI:** A clean and responsive user interface built entirely in Python using Streamlit.
- **Multi-Language Support:** Supports translation between English, Spanish, French, German, Hindi, and Chinese (Simplified).
- **Auto-Detection:** Automatically detects the source language of the inputted text.
- **Instant Translation:** Powered by the `googletrans` library for fast and accurate translations.

## 🛠️ Technologies Used
- Python 3.x
- `streamlit` (for the web interface)
- `googletrans` (for accessing the Google Translate API)

## 🚀 How to Run

1. **Install Dependencies:**
   Ensure you have the required libraries installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   Start the Streamlit server:
   ```bash
   streamlit run main.py
   ```

3. **Usage:**
   - Open the local URL provided by Streamlit in your web browser.
   - Enter the text you wish to translate.
   - Select the Source and Target languages from the dropdown menus.
   - Click "Translate" to view the result.
