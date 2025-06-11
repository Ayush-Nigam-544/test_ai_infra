# RAG API Client

This project is a Python API application that connects to the BentoML service endpoints for serving a Retrieval-Augmented Generation (RAG) model. It utilizes FastAPI for building the API and includes various components for managing requests and responses.

## Project Structure

```
rag-api-client
├── src
│   ├── main.py               # Entry point of the application
│   ├── api                   # Contains API routes and models
│   │   ├── __init__.py
│   │   ├── routes.py         # Defines API routes for interacting with the RAG model
│   │   └── models.py         # Pydantic models for request and response validation
│   ├── clients                # Contains client logic for connecting to BentoML service
│   │   ├── __init__.py
│   │   └── bentoml_client.py  # Functions for sending requests to the RAG application
│   ├── config                 # Configuration settings for the application
│   │   ├── __init__.py
│   │   └── settings.py        # API URLs and environment variables
│   └── utils                  # Utility functions for the application
│       ├── __init__.py
│       └── validators.py      # Functions for validating input data
├── tests                      # Contains unit tests for the application
│   ├── __init__.py
│   ├── test_api.py           # Unit tests for API routes
│   └── test_client.py        # Unit tests for the BentoML client
├── requirements.txt           # Lists project dependencies
├── .env.example               # Example environment variables
├── .gitignore                 # Files and directories to ignore by Git
└── README.md                  # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd rag-api-client
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Copy `.env.example` to `.env` and fill in the required values.

## Usage

To run the application, execute the following command:

```
uvicorn src.main:app --reload
```

This will start the FastAPI server, and you can access the API documentation at `http://127.0.0.1:8000/docs`.

## API Endpoints

- **GET /health**: Check the health status of the service.
- **POST /rag_query**: Send a query to the RAG model and receive a response.

## Testing

To run the tests, use the following command:

```
pytest
```

This will execute all unit tests defined in the `tests` directory.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.