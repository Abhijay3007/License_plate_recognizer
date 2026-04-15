import io
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


DEFAULT_DB_PATH = Path("features") / "session_logs.db"


def init_session_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT,
                normalized_plate TEXT,
                timestamp TEXT,
                detection_confidence REAL,
                validation_status TEXT,
                validation_reason TEXT,
                lookup_alert TEXT,
                speed_kmph REAL,
                image_name TEXT
            )
            """
        )
        connection.commit()


def log_detection(
    plate_text: str,
    normalized_plate: str,
    detection_confidence: float,
    validation: Dict,
    lookup: Dict,
    image_name: str,
    speed_kmph: Optional[float] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    init_session_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO detection_logs (
                plate_text, normalized_plate, timestamp, detection_confidence,
                validation_status, validation_reason, lookup_alert, speed_kmph, image_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plate_text,
                normalized_plate,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                float(detection_confidence),
                validation.get("status"),
                validation.get("reason"),
                lookup.get("alert_level"),
                speed_kmph,
                image_name,
            ),
        )
        connection.commit()


def fetch_logs(db_path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    init_session_db(db_path)
    with sqlite3.connect(db_path) as connection:
        dataframe = pd.read_sql_query(
            "SELECT * FROM detection_logs ORDER BY id DESC",
            connection,
        )
    return dataframe


def export_logs_csv(db_path: Path | str = DEFAULT_DB_PATH) -> bytes:
    dataframe = fetch_logs(db_path)
    return dataframe.to_csv(index=False).encode("utf-8")


def export_logs_pdf(db_path: Path | str = DEFAULT_DB_PATH) -> bytes:
    dataframe = fetch_logs(db_path)
    buffer = io.BytesIO()

    with PdfPages(buffer) as pdf:
        if dataframe.empty:
            figure, axis = plt.subplots(figsize=(11.69, 8.27))
            axis.axis("off")
            axis.text(0.5, 0.5, "No session logs available yet.", ha="center", va="center", fontsize=18)
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
        else:
            rows_per_page = 18
            for start_index in range(0, len(dataframe), rows_per_page):
                chunk = dataframe.iloc[start_index : start_index + rows_per_page]
                figure, axis = plt.subplots(figsize=(14, 8))
                axis.axis("off")
                axis.set_title("ANPR Session Report", fontsize=16, pad=18)
                table = axis.table(
                    cellText=chunk.values,
                    colLabels=chunk.columns,
                    loc="center",
                    cellLoc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.4)
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)

    buffer.seek(0)
    return buffer.getvalue()
