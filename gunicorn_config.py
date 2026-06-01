def post_fork(server, worker):
    import app.rag_pipeline as _rag
    _rag._collection = None
    print(f"[gunicorn] post_fork: reset ChromaDB for worker {worker.pid}", flush=True)
