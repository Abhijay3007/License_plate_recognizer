# Vehicle License Plate Recognition with YOLOv5 + OCR

This project detects a vehicle number plate with **YOLOv5**, extracts the text with **EasyOCR/Tesseract-style OCR tooling**, and now includes extra ML-system features that make it stronger as a semester project:

- Plate format validation for fake or tampered plate flagging
- Local watchlist lookup to simulate a traffic surveillance alert system
- Dual-frame speed estimation from plate movement
- Detection confidence heatmap export
- SQLite session logging with CSV/PDF export

## Project Structure

```text
features/
  data/
    vehicle_watchlist.json
  heatmap.py
  pipeline.py
  plate_lookup.py
  plate_validation.py
  session_logger.py
  speed_estimation.py
streamlit_app.py
main.py
app.py
```

## Install

```bash
pip install -r requirements.txt
```

## Run the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

## Streamlit Workflow

1. Upload one vehicle image to run plate detection and OCR.
2. Review the recognized plate, validation result, watchlist alert, and heatmap.
3. Upload two frames of the same vehicle for speed estimation.
4. Export the session log as CSV or PDF from the dashboard.

## New Feature Summary

### 1. Fake / Tampered Plate Flagging

`features/plate_validation.py` checks OCR text against Indian registration rules using regex and basic structural checks. It labels results as:

- `VALID`
- `SUSPICIOUS`
- `INVALID`

### 2. Plate Lookup Simulation

`features/plate_lookup.py` searches a local JSON watchlist:

- stolen vehicles
- flagged owners
- expired registration

Sample records live in `features/data/vehicle_watchlist.json`.

### 3. Speed Estimation

`features/speed_estimation.py` compares plate center movement between two frames and converts displacement into approximate speed using:

```text
speed = pixel_shift × meters_per_pixel ÷ time_interval
```

### 4. Detection Confidence Heatmap

`features/heatmap.py` creates a confidence-focused overlay from YOLO detections and saves the annotated output in `exports/heatmaps/`.

### 5. Session Logging and Export

`features/session_logger.py` stores every analysis result in SQLite and exports logs as:

- CSV
- PDF

## Notes

- Best results come from clear front or rear vehicle images.
- For speed estimation, use a fixed camera and a realistic calibration factor.
- The included watchlist is only a local simulation for demonstration.
