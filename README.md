# Python RAG Agent with LLM

## Project Overview
This is a Retrieval-Augmented Generation (RAG) agent built using Python, leveraging LangChain and OpenAI's language models to create an intelligent document interaction and querying system.

## Prerequisites
- Python 3.8+
- pip

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zsantana/python-rag-agent-llm.git
cd python-rag-agent-llm
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root and add:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Dependencies
- Streamlit: Web application framework
- python-dotenv: Environment variable management
- LangChain: AI/LLM integration framework
- OpenAI: Language model API

## Running the Application
```bash
streamlit run app.py
```

## Project Structure
- `requirements.txt`: Project dependencies
- `app.py`: Main Streamlit application
- `.env`: Environment configuration (not tracked in version control)

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License
[Specify your license here]

## Contact
[Your contact information]
