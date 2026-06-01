import os
from typing import Any

import httpx
from flask import Flask, jsonify, request, render_template
from werkzeug.exceptions import HTTPException

from app.rag_pipeline import answer_question

app = Flask(__name__)


@app.errorhandler(404)
def not_found(e: HTTPException):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e: HTTPException):
    return jsonify({"error": "Method not allowed."}), 405


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/chat", methods=["POST"])
def chat():
    data: dict[str, Any] = request.get_json(silent=True) or {}
    question: str = str(data.get("question", "")).strip()

    if not question:
        return jsonify({
            "answer": "Please enter a question.",
            "citations": [],
            "snippets": []
        }), 400

    try:
        result = answer_question(question)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except httpx.TimeoutException:
        return jsonify({"error": "LLM request timed out."}), 504
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        detail = e.response.text
        return jsonify({"error": f"LLM API error: {status}", "detail": detail}), 502
    except Exception as e:
        return jsonify({
            "answer": "Sorry, something went wrong while generating the answer.",
            "error": str(e),
            "citations": [],
            "snippets": []
        }), 500


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
