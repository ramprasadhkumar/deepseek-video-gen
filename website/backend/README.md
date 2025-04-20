# DeepSeek Video Generator Backend

This is the FastAPI backend for the DeepSeek Video Generator application. It handles model loading, code generation, and video creation using Manim.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set the environment variable for the model path (optional):

```bash
export VIDEOGEN_MODEL_PATH="path/to/your/model.pt"
```

## Running the Backend

Start the FastAPI server with:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at http://localhost:8000

## API Endpoints

- `POST /generate`: Generate a video from a text prompt
- `GET /video`: Get the latest generated video
- `GET /status`: Check if the server is running

## Fallback Mode

If the DeepSeek model is not available, the backend will use a fallback mechanism that returns a pre-generated Pythagorean Theorem video and code.
