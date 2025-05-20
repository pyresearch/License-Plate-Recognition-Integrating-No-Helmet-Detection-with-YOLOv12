import cv2
import supervision as sv
from ultralytics import YOLO

# Define the path to the weights file
# Load the model
model = YOLO("lprbest.pt")

def process_webcam():
    cap = cv2.VideoCapture("demo.mp4")  # Replace with 0 for the default webcam

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Display the annotated frame
        cv2.imshow("Webcam", annotated_frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting webcam processing...")
    process_webcam()