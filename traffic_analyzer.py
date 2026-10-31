
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from ultralytics import YOLO

# ---------------------------------
# 1. VIDEO SETTINGS
# ---------------------------------

video_path = r"C:\Users\shaha\OneDrive\Desktop\car video.mp4"

output_path = r"C:\Users\shaha\OneDrive\Desktop\traffic_detected.mp4"

CLIP_DURATION = 60

# ---------------------------------
# 2. LOAD YOLO MODEL
# ---------------------------------

print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("YOLO model loaded!")

# COCO class IDs:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

vehicle_classes = [2, 3, 5, 7]

# ---------------------------------
# 3. OPEN VIDEO
# ---------------------------------

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0 or total_frames <= 0:
    print("Invalid video information!")
    cap.release()
    exit()

duration = total_frames / fps

print("Video duration:", round(duration / 60, 2), "minutes")

# ---------------------------------
# 4. SELECT 60-SECOND CLIP
# ---------------------------------

clip_duration = min(CLIP_DURATION, duration)

start_time = max(0, (duration - clip_duration) / 2)

start_frame = int(start_time * fps)
end_frame = min(
    total_frames,
    int((start_time + clip_duration) * fps)
)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    print("Could not create output video!")
    cap.release()
    exit()

# ---------------------------------
# 5. INITIALIZE COUNTERS
# ---------------------------------

vehicle_counts = {
    "car": 0,
    "motorcycle": 0,
    "bus": 0,
    "truck": 0
}

seen_ids = set()

motion_values = []
time_values = []

previous_gray = None

frame_number = start_frame

print("\nStarting vehicle detection...")
print("Press Q to stop.")

# ---------------------------------
# 6. PROCESS VIDEO
# ---------------------------------

while frame_number < end_frame:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO vehicle detection and tracking
    results = model.track(
        frame,
        persist=True,
        classes=vehicle_classes,
        conf=0.35,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    # Draw bounding boxes and labels
    annotated_frame = result.plot()

    current_frame_count = 0

    # ---------------------------------
    # 7. COUNT UNIQUE TRACKED VEHICLES
    # ---------------------------------

    if result.boxes is not None:

        boxes = result.boxes

        if boxes.id is not None:

            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()

            for track_id, class_id in zip(track_ids, class_ids):

                current_frame_count += 1

                # Count each tracking ID only once
                if track_id not in seen_ids:

                    seen_ids.add(track_id)

                    vehicle_name = model.names[class_id]

                    if vehicle_name in vehicle_counts:
                        vehicle_counts[vehicle_name] += 1

    # ---------------------------------
    # 8. MOTION ANALYSIS USING OPENCV
    # ---------------------------------

    small_frame = cv2.resize(frame, (320, 180))

    gray = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2GRAY
    )

    motion = 0

    if previous_gray is not None:

        difference = cv2.absdiff(
            previous_gray,
            gray
        )

        motion = float(np.mean(difference))

        motion_values.append(motion)

        elapsed = (frame_number - start_frame) / fps

        time_values.append(elapsed)

    previous_gray = gray

    # ---------------------------------
    # 9. DISPLAY LIVE DASHBOARD
    # ---------------------------------

    total_detected = len(seen_ids)

    cv2.putText(
        annotated_frame,
        "SMART TRAFFIC INTELLIGENCE",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Vehicles in frame: {current_frame_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Unique vehicles tracked: {total_detected}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Display vehicle-type counts
    y_position = 145

    for name, count in vehicle_counts.items():

        text = f"{name.capitalize()}: {count}"

        cv2.putText(
            annotated_frame,
            text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        y_position += 30

    # Display motion
    cv2.putText(
        annotated_frame,
        f"Motion: {motion:.2f}",
        (20, y_position + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )

    # ---------------------------------
    # 10. SHOW AND SAVE VIDEO
    # ---------------------------------

    cv2.imshow(
        "Pakistan Smart Traffic Intelligence",
        annotated_frame
    )

    writer.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user.")
        break

    frame_number += 1

# ---------------------------------
# 11. RELEASE RESOURCES
# ---------------------------------

cap.release()
writer.release()
cv2.destroyAllWindows()

# ---------------------------------
# 12. FINAL TRAFFIC REPORT
# ---------------------------------

print("\n========== TRAFFIC REPORT ==========")

for name, count in vehicle_counts.items():
    print(name.capitalize(), ":", count)

print("Total unique tracked IDs:", len(seen_ids))

print("Annotated video saved at:")
print(output_path)

# ---------------------------------
# 13. MATPLOTLIB GRAPH
# ---------------------------------

if len(motion_values) > 0:

    motion_array = np.array(motion_values)

    print(
        "Average motion:",
        round(float(np.mean(motion_array)), 2)
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        time_values,
        motion_array,
        label="Traffic Motion"
    )

    plt.title("Traffic Motion Analysis")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Pixel Difference")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        r"C:\Users\shaha\OneDrive\Desktop\traffic_motion_graph.png"
    )

    plt.show()

# ---------------------------------
# 14. VEHICLE DISTRIBUTION GRAPH
# ---------------------------------

names = list(vehicle_counts.keys())
counts = list(vehicle_counts.values())

plt.figure(figsize=(9, 5))

plt.bar(names, counts)

plt.title("Vehicle Type Distribution")
plt.xlabel("Vehicle Type")
plt.ylabel("Unique Tracked IDs")

plt.tight_layout()

plt.savefig(
    r"C:\Users\shaha\OneDrive\Desktop\vehicle_distribution.png"
)

plt.show()

print("\nSTAGE 2 COMPLETED!")
