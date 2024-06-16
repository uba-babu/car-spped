import cv2
import pandas as pd
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Define the mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Set up the named window and mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video capture
cap = cv2.VideoCapture(r'C:\Users\U.B.A Yadav\OneDrive\Desktop\speed\highway_mini.mp4')

# Load the class names
with open(r'C:\Users\U.B.A Yadav\OneDrive\Desktop\speed\coco.txt', "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize variables
count = 0
trackers = []

cy1 = 322
cy2 = 368
offset = 6

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            # Initialize the tracker for each detected car
            tracker = cv2.TrackerKCF_create()
            bbox = (x1, y1, x2 - x1, y2 - y1)
            tracker.init(frame, bbox)
            trackers.append(tracker)

    for tracker in trackers:
        ret, bbox = tracker.update(frame)
        if ret:
            x3, y3, w, h = [int(v) for v in bbox]
            x4, y4 = x3 + w, y3 + h
            cx = int((x3 + x4) / 2)
            cy = int((y3 + y4) / 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(trackers.index(tracker)), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # cv2.line(frame,(274,cy1),(814,cy1),(255,255,255),1)
    # cv2.line(frame,(177,cy2),(927,cy2),(255,255,255),1)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()



count = 0

tracker = legacy.TrackerKCF_create()

cy1 = 322
cy2 = 368
offset = 6

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    # print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    # print(px)
    list = []

    for index, row in px.iterrows():
        # print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # cv2.line(frame,(274,cy1),(814,cy1),(255,255,255),1)
    # cv2.line(frame,(177,cy2),(927,cy2),(255,255,255),1)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

