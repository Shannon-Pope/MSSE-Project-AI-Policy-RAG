"""
Pre-download the ONNX embedding model during the build phase.

This runs before ingest.py so the model is guaranteed to be on disk
inside the project directory (preserved in Render's build snapshot).
At runtime, _download_model_if_not_exists() finds it immediately and
skips the S3 download.
"""
from pathlib import Path
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

_ONNX_CACHE = Path(__file__).resolve().parent.parent / "vectorstore" / "onnx_cache"
ONNXMiniLM_L6_V2.DOWNLOAD_PATH = _ONNX_CACHE / ONNXMiniLM_L6_V2.MODEL_NAME

ef = ONNXMiniLM_L6_V2()
print(f"ONNX download path: {ONNXMiniLM_L6_V2.DOWNLOAD_PATH}")

onnx_model = Path(str(ONNXMiniLM_L6_V2.DOWNLOAD_PATH)) / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME / "model.onnx"
if onnx_model.exists():
    print(f"ONNX model already present at {onnx_model}")
else:
    print("Downloading ONNX model from S3 ...")
    ef._download_model_if_not_exists()
    print(f"Download complete. model.onnx exists: {onnx_model.exists()}")
