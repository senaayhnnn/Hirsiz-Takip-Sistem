import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO(r"C:\Users\pc\Downloads\yolo11n-pose.pt")
cap = cv2.VideoCapture(r"C:\Users\pc\Downloads\Robbery gone wrong🤭🤭🤭 #robbery #shopping #funny #shorts (online-video-cutter.com).mp4")

tolerance = 30
thief_box = None  # Hırsızın kutusu (x1, y1, x2, y2)

keypoint_names = [
    "nose", "left eye", "right eye", "left ear", "right ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left hip", "right hip",
    "left knee", "right knee", "left ankle", "right ankle"
]

pose_pairs = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12), (5, 11), (6, 12)
]

start_point = (99, 372)
end_point = (605, 164)
line_y = start_point[1]

def iou(box1, box2):
    """IoU hesapla: box = (x1, y1, x2, y2)"""
    xa1, ya1, xa2, ya2 = box1
    xb1, yb1, xb2, yb2 = box2

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (xa2 - xa1) * (ya2 - ya1)
    area2 = (xb2 - xb1) * (yb2 - yb1)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def click_event(event, x, y, flags, param):
    global thief_box
    if event == cv2.EVENT_LBUTTONDOWN:
        thief_box = (x-20, y-20, x+20, y+20)
        print(f"[INFO] Hırsız kutusu seçildi: {thief_box}")

cv2.namedWindow("Hirsizlik Tespiti")
cv2.setMouseCallback("Hirsizlik Tespiti", click_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    action_detected = False

    results = model.predict(source=frame, conf=0.5, verbose=False)[0]

    if results.keypoints is not None and results.boxes is not None:
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_box = (x1, y1, x2, y2)

            
            if thief_box is not None and iou(current_box, thief_box) < 0.3:
                continue

            kp_all = results.keypoints.data[i].cpu().numpy()
            valid_points = kp_all[:, 2] > 0.5
            if np.sum(valid_points) == 0:
                continue
            avg_y = np.mean(kp_all[valid_points, 1])

            if avg_y < line_y:
                for idx, (x, y, conf) in enumerate(kp_all):
                    if conf > 0.8:
                        x, y = int(x), int(y)
                        name = keypoint_names[idx]
                        cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
                        cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                for partA, partB in pose_pairs:
                    if kp_all[partA][2] > 0.5 and kp_all[partB][2] > 0.5:
                        xA, yA = int(kp_all[partA][0]), int(kp_all[partA][1])
                        xB, yB = int(kp_all[partB][0]), int(kp_all[partB][1])
                        cv2.line(frame, (xA, yA), (xB, yB), (0, 255, 255), 2)

                
                for joint_name in ["right wrist", "right elbow", "left wrist", "left elbow"]:
                    idx = keypoint_names.index(joint_name)
                    y = kp_all[idx][1]
                    conf = kp_all[idx][2]
                    if conf > 0.8 and y > line_y + tolerance:
                        action_detected = True
                        break

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    
    if action_detected:
        cv2.putText(frame, "Action Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    elif thief_box:
        cv2.putText(frame, "SAFE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "TIKLA -> HIRSIZ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)

    cv2.imshow("Hirsizlik Tespiti", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
