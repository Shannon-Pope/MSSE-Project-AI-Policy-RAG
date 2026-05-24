# AI Policy RAG Application

A Retrieval-Augmented Generation (RAG) application that answers questions about company policies and procedures.

## Features

- Policy document ingestion
- Vector search
- LLM-powered answers
- Citation support
- Flask web application

## Setup

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Mac/Linux:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create `.env` from `.env.example`

### Run Application

```bash
python app/app.py
```

## Health Check

```text
http://127.0.0.1:5000/health
```