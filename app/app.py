from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "AI Policy RAG App Running"

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(debug=True)