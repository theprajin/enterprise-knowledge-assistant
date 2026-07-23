# Enterprise Knowledge Assistant

An AI-powered assistant built with LangChain and Langfuse, designed to answer questions about your enterprise knowledge base.

## Features

- **Question Answering**: Get intelligent answers to your questions based on your knowledge base.
- **Traceability**: Track every question and answer through Langfuse for complete observability.
- **Prompt Management**: Manage and version prompts directly from the Langfuse UI.
- **Health Monitoring**: Built-in health checks for Langfuse connectivity.

## Prerequisites

- Python 3.9+
- Node.js (for frontend development)
- PostgreSQL database

## Setup

### 1. Environment Configuration

Copy the `.env.example` file to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

**Required Environment Variables:**

```env
LANGFUSE_PUBLIC_KEY=lf_pub_...        # Required for Langfuse observability
LANGFUSE_SECRET_KEY=lf_sec_...      # Required for Langfuse observability
LANGFUSE_BASE_URL=http://langfuse-web:3000  # Langfuse UI URL

DB_HOST=localhost
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=enterprise_knowledge
PORT=8000

# LLM Provider Configuration
OPENROUTER_API_KEY=sk-or-...        # Required for LLM calls
```

> **Note:** Langfuse is optional. If you don't have credentials, you can leave the `LANGFUSE_*` variables empty, and the application will run in a reduced-feature mode.

### 2. Install Dependencies

**Backend (Python)**:
```bash
pip install -r requirements.txt
```

### 3. Database Setup

Ensure your PostgreSQL database is running and accessible. The database will be created automatically if it doesn't exist.

### 4. Run the Application

**Backend:**
```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

## Usage

### Asking Questions

You can ask questions using the API or the Swagger UI.

**API Example:**
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the company's remote work policy?"}'
```

### Langfuse Integration

To use Langfuse:
1. Get your API keys from [Langfuse](https://langfuse.com).
2. Set the `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` environment variables.
3. Run the application.

All queries and responses will be automatically traced in Langfuse.

## Health Checks

You can check the health of the application and its connection to Langfuse:

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```