import cv2
from ultralytics import YOLO
import numpy as np

def point_line_distance(px, py, x1, y1, x2, y2):
    numerator = abs((y2 - y1)*px - (x2 - x1)*py + x2*y1 - y2*x1)
    denominator = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
    if denominator == 0:
        return float('inf')
    return numerator / denominator

model = YOLO(r"C:\Users\pc\Downloads\yolo11n-pose.pt")
cap = cv2.VideoCapture(r"C:\Users\pc\Downloads\Robbery gone wrong🤭🤭🤭 #robbery #shopping #funny #shorts (online-video-cutter.com).mp4")

tolerance = 30  

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

start_point = (99, 372)
end_point = (605, 164)
tezgah_distance_threshold =15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_width = frame.shape[1]
    mid_x = frame_width // 2

    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    action_detected = False

    results = model.predict(source=frame, conf=0.5, verbose=False)[0]

    if results.keypoints is not None and results.boxes is not None:
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_center_x = (x1 + x2) // 2

            if bbox_center_x > mid_x:
                continue

            kp_all = results.keypoints.data[i].cpu().numpy()

            valid_points = kp_all[:, 2] > 0.5
            if np.sum(valid_points) == 0:
                continue

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

           
            for joint_name in ["right wrist", "left wrist","right elbow","left elbow"]:
                idx = keypoint_names.index(joint_name)
                x, y, conf = kp_all[idx]
                if conf > 0.8:
                    x, y = int(x), int(y)
                    dist = point_line_distance(x, y, start_point[0], start_point[1], end_point[0], end_point[1])
                    if dist < tezgah_distance_threshold:
                        action_detected = True
                        break

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    
    if action_detected:
        cv2.putText(frame, "Action Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "SAFE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Hirsizlik Tespiti", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
