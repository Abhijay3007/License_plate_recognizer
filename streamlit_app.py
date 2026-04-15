from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from features.pipeline import analyze_image
from features.session_logger import export_logs_csv, export_logs_pdf, fetch_logs, log_detection
from features.speed_estimation import draw_speed_visualization, estimate_vehicle_speed


st.set_page_config(page_title="ANPR Project Dashboard", page_icon="C", layout="wide")
st.title("Vehicle License Plate Recognition")
st.caption("YOLOv5 plate detection + OCR + fraud flagging + watchlist alerts + speed estimation")


@st.cache_data(show_spinner="Running AI Analysis (Cached)...")
def cached_analyze(file_bytes: bytes, file_name: str):
    np_arr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return analyze_image(frame, image_name=file_name)


with st.sidebar:
    st.header("Project Settings")
    time_interval_seconds = st.number_input(
        "Dual-frame time gap (seconds)",
        min_value=0.1,
        value=1.0,
        step=0.1,
    )
    meters_per_pixel = st.number_input(
        "Calibration factor (meters per pixel)",
        min_value=0.001,
        value=0.05,
        step=0.005,
        format="%.3f",
    )
    st.markdown("Use the calibration factor from your camera setup for better speed estimates.")

single_image_file = st.file_uploader(
    "Upload a vehicle image for plate analysis",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

st.divider()
st.subheader("Dual-Frame Speed Estimation")
col_a, col_b = st.columns(2)
with col_a:
    first_frame_file = st.file_uploader(
        "First frame",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="frame_1",
    )
with col_b:
    second_frame_file = st.file_uploader(
        "Second frame",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="frame_2",
    )

if single_image_file is not None:
    results = cached_analyze(single_image_file.getvalue(), single_image_file.name)
    top_detection = results["top_detection"]
    confidence = top_detection["confidence"] if top_detection else 0.0

    upload_signature = (
        single_image_file.name,
        len(single_image_file.getvalue()),
        results["plate_text"],
    )
    if st.session_state.get("last_logged_upload") != upload_signature:
        log_detection(
            plate_text=results["plate_text"],
            normalized_plate=results["validation"]["normalized_text"],
            detection_confidence=confidence,
            validation=results["validation"],
            lookup=results["lookup"],
            image_name=single_image_file.name,
        )
        st.session_state["last_logged_upload"] = upload_signature

    overview_col, feature_col = st.columns([1.1, 1.2])

    with overview_col:
        st.subheader("Detection Result")
        st.image(
            cv2.cvtColor(results["annotated_image"], cv2.COLOR_BGR2RGB),
            caption="YOLOv5 plate detection",
            use_container_width=True,
        )

        if results["plate_crop"] is not None:
            st.image(
                cv2.cvtColor(results["plate_crop"], cv2.COLOR_BGR2RGB),
                caption="Detected plate crop",
                width=360,
            )

    with feature_col:
        st.subheader("Recognized Plate")
        st.metric("Best OCR Output", results["plate_text"])
        if results["candidates"]:
            st.write("Top OCR candidates:", ", ".join(results["candidates"]))

        validation = results["validation"]
        status_color = {
            "VALID": "green",
            "SUSPICIOUS": "orange",
            "INVALID": "red",
        }.get(validation["status"], "gray")
        st.markdown(
            f"**Format Validation:** :{status_color}[{validation['status']}]  \n{validation['reason']}"
        )

        lookup = results["lookup"]
        if lookup["match_found"]:
            st.error("ALERT: Plate matched the local watchlist database.")
            st.write("Reasons:", "; ".join(lookup["reasons"]))
            if lookup["record"]:
                st.json(lookup["record"])
        else:
            st.success(lookup["reasons"][0])

        st.write(f"Detection confidence: `{confidence:.2f}`")
        if results["heatmap_path"]:
            st.caption(f"Heatmap saved to `{results['heatmap_path']}`")

    st.subheader("Detection Confidence Heatmap")
    st.image(
        cv2.cvtColor(results["heatmap_image"], cv2.COLOR_BGR2RGB),
        caption="Confidence-focused overlay",
        use_container_width=True,
    )

if first_frame_file is not None and second_frame_file is not None:
    first_results = cached_analyze(first_frame_file.getvalue(), first_frame_file.name)
    second_results = cached_analyze(second_frame_file.getvalue(), second_frame_file.name)
    speed_result = estimate_vehicle_speed(
        first_results["top_detection"],
        second_results["top_detection"],
        time_interval_seconds=time_interval_seconds,
        meters_per_pixel=meters_per_pixel,
    )

    st.subheader("Speed Estimation Output")
    dual_left, dual_right = st.columns(2)
    with dual_left:
        st.image(
            cv2.cvtColor(first_results["annotated_image"], cv2.COLOR_BGR2RGB),
            caption="First frame",
            use_container_width=True,
        )
    with dual_right:
        if speed_result["success"]:
            speed_visual = draw_speed_visualization(
                second_results["annotated_image"],
                first_results["top_detection"],
                second_results["top_detection"],
                speed_result,
            )
            st.image(
                cv2.cvtColor(speed_visual, cv2.COLOR_BGR2RGB),
                caption="Second frame with motion arrow",
                use_container_width=True,
            )
            st.metric("Approx. speed", f"{speed_result['speed_kmph']} km/h")
            st.write(
                f"Pixel shift: {speed_result['pixel_shift']} px | "
                f"Distance: {speed_result['distance_meters']} m"
            )
        else:
            st.warning(speed_result["reason"])
            st.image(
                cv2.cvtColor(second_results["annotated_image"], cv2.COLOR_BGR2RGB),
                caption="Second frame",
                use_container_width=True,
            )

st.divider()
st.subheader("Session Log")
log_df = fetch_logs()
st.dataframe(log_df, use_container_width=True, hide_index=True)

csv_bytes = export_logs_csv()
pdf_bytes = export_logs_pdf()
export_col_1, export_col_2 = st.columns(2)
with export_col_1:
    st.download_button(
        "Export CSV Report",
        data=csv_bytes,
        file_name="anpr_session_report.csv",
        mime="text/csv",
    )
with export_col_2:
    st.download_button(
        "Export PDF Report",
        data=pdf_bytes,
        file_name="anpr_session_report.pdf",
        mime="application/pdf",
    )

st.caption(
    "Tip: for the speed module, use two images from the same fixed camera and set the real time gap correctly."
)
