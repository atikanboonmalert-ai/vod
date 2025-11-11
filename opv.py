from ultralytics import YOLO
import cv2

# โหลดโมเดล YOLOv8
model = YOLO("yolov8n.pt")

# เปิดไฟล์วิดีโอ (หรือใช้กล้องเว็บแคมได้โดยใส่เลข 0)
video_path = "8888.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ ไม่สามารถเปิดวิดีโอได้")
    exit()

# อ่านชื่อคลาสทั้งหมด
names = model.names

while True:
    ret, frame = cap.read()
    if not ret:
        break  # จบวิดีโอ

    # ตรวจจับวัตถุในเฟรม
    results = model(frame)
    detections = results[0].boxes.data

    car_count = 0

    # วาดกรอบรอบรถ
    for box in detections:
        x1, y1, x2, y2, conf, cls = box
        label = names[int(cls)]
        if label in ["car", "truck", "bus", "motorbike"]:
            car_count += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # แสดงจำนวนรถที่ตรวจพบในเฟรม
    cv2.putText(frame, f"Detected vehicles: {car_count} คัน",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # แสดงผลเฟรม
    cv2.imshow("Vehicle Detection Result", frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
