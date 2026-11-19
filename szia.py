from pathlib import Path
import cv2
import yt_dlp
from ultralytics import YOLO

YOUTUBE_URL = "https://www.youtube.com/watch?v=zMCea32gpmg"

ydl_opts = {
    "format": "bestvideo*",
    "quiet": False,
    "js_runtimes": {
        "bun": {}
    },
    "remote_components": ["ejs:npm"]
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(YOUTUBE_URL, download=False)
    stream_url = info["url"]

model = YOLO("yolo26l.pt")

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Nem sikerült megnyitni a streamet!")
    exit()

print("A stream elindult.")

LINES = [
    {
        "name": "1. vonal (Zebra)",
        "start": (1230, 390),
        "end": (1200, 450),
        "color": (0, 255, 255),
        "counted_ids": set()
    },
    {
        "name": "2. vonal (Jarda)",
        "start": (1150, 650),
        "end": (1600, 650),
        "color": (255, 255, 0),
        "counted_ids": set()
    }
]

track_history = {}  # track_id -> előző (x, y) talppont

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
# ------------------------------------------------------------------

while True:
    ret, frame = cap.read()

    if not ret:
        print("Nem érkezik több képkocka.")
        break

    results = model.track(
        frame,
        persist=True,
        tracker=str(Path(__file__).parent / "custom_botsort.yaml"),
        classes=[0, 1],
        conf=0.15,
        iou=0.5,
        imgsz=1536,
        device=0,
        verbose=False
    )

    annotated_frame = results[0].plot()

    for line in LINES:
        cv2.line(annotated_frame, line["start"], line["end"], line["color"], 3)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, classes):
            if cls != 0:
                continue

            x1, y1, x2, y2 = box
            curr_pos = (int((x1 + x2) / 2), int(y2))  # Talppont

            if track_id in track_history:
                prev_pos = track_history[track_id]
                for line in LINES:
                    if track_id not in line["counted_ids"] and intersect(prev_pos, curr_pos, line["start"], line["end"]):
                        line["counted_ids"].add(track_id)

            track_history[track_id] = curr_pos

            is_counted = any(track_id in line["counted_ids"] for line in LINES)
            dot_color = (0, 255, 0) if is_counted else (0, 0, 255)
            cv2.circle(annotated_frame, curr_pos, 4, dot_color, -1)

    # Számlálók kiírása a bal felső sarokba
    for i, line in enumerate(LINES):
        cv2.putText(
            annotated_frame,
            f"{line['name']}: {len(line['counted_ids'])} fo",
            (30, 50 + i * 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            line["color"],
            3
        )

    cv2.imshow("Szia- AI Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()