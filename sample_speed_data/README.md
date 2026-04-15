# Sample Speed Data

This folder contains ready-to-use image pairs for testing the dual-frame speed estimation feature in `streamlit_app.py`.

## Included Pairs

- `pair_1/frame_1.jpg`
- `pair_1/frame_2.jpg`
- `pair_2/frame_1.jpg`
- `pair_2/frame_2.jpg`
- `pair_3/frame_1.jpg`
- `pair_3/frame_2.jpg`

Each pair also includes a `config.txt` file with the suggested values to use in the Streamlit sidebar:

- `time_interval_seconds`
- `meters_per_pixel`

## How to Test

1. Run `streamlit run streamlit_app.py`
2. Open the `Dual-Frame Speed Estimation` section
3. Upload `frame_1.jpg` and `frame_2.jpg` from the same pair
4. Enter the `time_interval_seconds` value from that pair's `config.txt`
5. Enter the `meters_per_pixel` value from that pair's `config.txt`
6. Read the estimated speed output

## Note

These are synthetic demo pairs created by shifting the same base vehicle image slightly. They are useful for testing the pipeline and UI behavior, but they are not physically accurate traffic-camera measurements.
