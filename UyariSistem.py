import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO(r"C:\Users\pc\Downloads\yolo11n-pose.pt")
cap = cv2.VideoCapture(0)

cizgi = 0.5  
status = "Default"
prev_frame = None

keypoint_names = [
    "nose", "left eye", "right eye", "left ear", "right ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left hip", "right hip",
    "left knee", "right knee", "left ankle", "right ankle"
]

pose_pairs = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
    (5, 11), (6, 12)
]

backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    linex = int(w * cizgi)

    cv2.line(frame, (linex, 0), (linex, h), (0, 255, 0), 2)

    results = model.predict(source=frame, conf=0.8, classes=[0], verbose=False)[0]

    if results.keypoints is not None and results.boxes is not None:
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2

            kp_all = results.keypoints.data[i].cpu().numpy() 

            for idx, (x, y, conf) in enumerate(kp_all):
                if conf > 0.8:  
                    x, y = int(x), int(y)
                    name = keypoint_names[idx]
                    cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
                    cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            for partA, partB in pose_pairs:
                if partA < len(kp_all) and partB < len(kp_all):
                    xA, yA, cA = kp_all[partA]
                    xB, yB, cB = kp_all[partB]
                    if cA > 0.5 and cB > 0.5:  
                        cv2.line(frame, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 255), 2)

            if cx > linex + 15:
                status = "Guvensiz Bolge - UYARI!"
                color = (0, 0, 255)
            elif cx < linex - 15:
                status = "Guvenli Bolge"
                color = (0, 255, 0)
            else:
                status = "Sinirda"
                color = (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "Person", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

  
    fgMask = backSub.apply(frame)

    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=2)
    fgMask = cv2.dilate(fgMask, kernel, iterations=2)

    motion_area = cv2.countNonZero(fgMask)
    if motion_area > 5000:
        cv2.putText(frame, "Hareket Algilandi!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Uyari Sistemi", frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
