import cv2
import numpy as np
from collections import deque

class ObjectTracker:
    def __init__(self, max_missing=30):
        self.next_object_id = 0
        self.objects = {}
        self.max_missing = max_missing

    def update(self, detections):
        updated_objects = {}
        assigned_ids = set()

        for obj_id, (bbox, missing_frames) in self.objects.items():
            best_match_idx = -1
            min_dist = float('inf')
            for i, new_bbox in enumerate(detections):
                dist = np.linalg.norm(np.array(bbox[:2]) - np.array(new_bbox[:2]))
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = i

            if best_match_idx != -1 and min_dist < 50:
                updated_objects[obj_id] = (detections[best_match_idx], 0)
                assigned_ids.add(best_match_idx)
            else:
                if missing_frames + 1 < self.max_missing:
                    updated_objects[obj_id] = (bbox, missing_frames + 1)

        for i, new_bbox in enumerate(detections):
            if i not in assigned_ids:
                updated_objects[self.next_object_id] = (new_bbox, 0)
                self.next_object_id += 1
        
        self.objects = updated_objects
        return [(obj_id, obj_data[0]) for obj_id, obj_data in self.objects.items() if obj_data[1] == 0]

def main():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    tracker = ObjectTracker()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream or file")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video stream")
            break

        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        detected_boxes = []
        class_ids = []
        confidences = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    detected_boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(detected_boxes, confidences, 0.5, 0.4)

        current_detections = []
        current_class_ids = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                current_detections.append(detected_boxes[i])
                current_class_ids.append(class_ids[i])
        
        tracked_objects = tracker.update(current_detections)

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        for obj_id, bbox in tracked_objects:
            x, y, w, h = bbox
            matched_class_id = -1
            for i, det_bbox in enumerate(current_detections):
                if np.array_equal(det_bbox, bbox) and i < len(current_class_ids):
                    matched_class_id = current_class_ids[i]
                    break
            
            if matched_class_id != -1:
                label = str(classes[matched_class_id])
                color = colors[matched_class_id]
            else:
                label = "Unknown"
                color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID: {obj_id} {label}", (x, y - 10), font, 1, color, 2)

        cv2.imshow('Object Detection and Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
