# ArXiv Paper Summarizer

This Streamlit application allows users to summarize arXiv papers by providing the arXiv ID. It fetches the paper details and generates a summary using the OpenAI API.

## Features

- Input arXiv ID to fetch paper details.
- Display paper title, authors, abstract, and other metadata.
- Generate a concise summary of the paper using a LLM.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/mike-whypred/paper-summarizer-single.git
    cd arxiv-paper-summarizer
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory and add your OpenAI API key:
    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Enter the arXiv ID of the paper you want to summarize and click "Summarize".

## File Structure

- `app.py`: The main Streamlit application file.
- `requirements.txt`: The list of dependencies required for the project.
- `.env`: File to store environment variables (e.g., OpenAI API key, Chatpdf key).

## Example

To summarize the paper with arXiv ID `1234.56789`, enter `1234.56789` in the input field and click "Summarize". The application will fetch the paper details and display a summary.

## Dependencies

- `arxiv`
- `pandas`
- `numpy`
- `python-dotenv`
- `requests`
- `streamlit`
- `openai`

## License

This project is licensed under the MIT License -
