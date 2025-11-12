import cv2
from ultralytics import YOLO
import time

# โหลดโมเดลระดับกลาง (แม่นกว่า yolov8n)
model = YOLO("yolov8m.pt")  # หรือเปลี่ยนเป็น yolov8l.pt สำหรับความแม่นยำสูงสุด

# เปิดเว็บแคม
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("ไม่สามารถเปิดเว็บแคมได้")

paused = False  # สถานะหยุด
fps_time = time.time()

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # ตรวจจับวัตถุ
        results = model(frame, conf=0.4)[0]  # conf=0.4 ปรับค่าความมั่นใจได้
        names = model.names
        car_count = 0

        # วาดกรอบเฉพาะรถ
        if hasattr(results, "boxes") and results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                if label.lower() == "car":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                    car_count += 1

        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"Cars: {car_count}  FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Car Detection", frame)
    else:
        # แสดงข้อความเมื่อหยุด
        paused_frame = frame.copy()
        cv2.putText(paused_frame, "PAUSED - press 's' to resume", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("YOLOv8 Car Detection", paused_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # ปุ่ม s = หยุด/เล่นต่อ
        paused = not paused
    elif key == 27 or key == ord('q'):  # ESC หรือ q เพื่อออก
        break

cap.release()
cv2.destroyAllWindows()
