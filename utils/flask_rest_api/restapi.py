# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing a YOLOv5s model with watchlist alert support
"""

import argparse
import io
import json
from pathlib import Path

import torch
from flask import Flask, request, jsonify
from PIL import Image

# Add project path to imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.plate_lookup import lookup_plate_record

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5s"
DETECTION_WITH_WATCHLIST_URL = "/v1/object-detection/yolov5s-with-watchlist"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return jsonify({"error": "Only POST requests are supported"}), 400

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        results = model(im, size=640)  # reduce size=320 for faster inference
        return results.pandas().xyxy[0].to_json(orient="records")
    
    return jsonify({"error": "No image file provided"}), 400


@app.route(DETECTION_WITH_WATCHLIST_URL, methods=["POST"])
def predict_with_watchlist():
    """Object detection with watchlist alert checking"""
    if not request.method == "POST":
        return jsonify({"error": "Only POST requests are supported"}), 400

    if not request.files.get("image"):
        return jsonify({"error": "No image file provided"}), 400
    
    try:
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        # Run object detection
        results = model(im, size=640)
        detections = results.pandas().xyxy[0].to_json(orient="records")
        
        # For demonstration, return detections with watchlist info
        # In a real scenario, you'd extract plate text from detections 
        # and check against watchlist
        return jsonify({
            "detections": json.loads(detections) if isinstance(detections, str) else detections,
            "watchlist_enabled": True,
            "message": "Extract plate text from detections and check against watchlist"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/v1/watchlist-check", methods=["POST"])
def check_watchlist():
    """Check a plate number against the watchlist database"""
    try:
        data = request.get_json()
        if not data or "plate_number" not in data:
            return jsonify({"error": "plate_number is required in request body"}), 400
        
        plate_number = data["plate_number"]
        watchlist_path = data.get("watchlist_path", Path("features") / "data" / "vehicle_watchlist.json")
        
        # Perform watchlist lookup
        result = lookup_plate_record(plate_number, watchlist_path)
        
        # Return comprehensive watchlist result
        return jsonify({
            "plate_number": result["plate_number"],
            "match_found": result["match_found"],
            "alert_level": result["alert_level"],
            "reasons": result["reasons"],
            "vehicle_record": result["record"] if result["match_found"] else None
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model with watchlist support")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    opt = parser.parse_args()

    # Fix known issue urllib.error.HTTPError 403: rate limit exceeded https://github.com/ultralytics/yolov5/pull/7210
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5s", force_reload=True
    )  # force_reload to recache
    app.run(host="0.0.0.0", port=opt.port)
