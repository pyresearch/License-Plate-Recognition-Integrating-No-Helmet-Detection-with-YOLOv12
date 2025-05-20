import cv2
import supervision as sv
from ultralytics import YOLO
from paddleocr import PaddleOCR
import typer
import numpy as np
import logging
import pyresearch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True)

# Load both models
plate_model = YOLO("lprbest.pt")  # Model for license plate detection
helmet_model = YOLO("best.pt")  # Model for helmet detection (With Helmet, Without Helmet)

app = typer.Typer()

def perform_ocr(image_array):
    """Perform OCR on an image array and return detected text."""
    if image_array is None or image_array.size == 0:
        logging.warning("Empty image provided for OCR")
        return ""

    try:
        results = ocr.ocr(image_array, rec=True)
        detected_text = []
        if results[0] is not None:
            for result in results[0]:
                text = result[1][0]
                detected_text.append(text)
        return ''.join(detected_text)
    except Exception as e:
        logging.error(f"OCR error: {e}")
        return ""

def process_webcam(output_file="output.mp4"):
    cap = cv2.VideoCapture("demo.mp4")  # Replace with 0 for webcam or use "demo.mp4" for helmet model compatibility

    if not cap.isOpened():
        logging.error("Could not open video file.")
        return

    # Get input video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"Input resolution: {width}x{height}, FPS: {fps}")

    # Reduce resolution if needed
    target_width = 1920
    target_height = 1080
    logging.info(f"Reducing resolution to: {target_width}x{target_height}")

    # Define codec and VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_file, fourcc, fps, (target_width, target_height))

    # Fallback to XVID if H264 fails
    if not out.isOpened():
        logging.warning("H264 codec failed, trying XVID with .avi...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_file = "output.avi"
        out = cv2.VideoWriter(output_file, fourcc, fps, (target_width, target_height))

    if not out.isOpened():
        logging.error("Could not initialize VideoWriter with any codec.")
        cap.release()
        return

    # Initialize supervision annotators for helmet detection
    helmet_box_annotator = sv.BoundingBoxAnnotator()
    helmet_label_annotator = sv.LabelAnnotator()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video stream")
            break

        frame_count += 1
        logging.info(f"Processing frame {frame_count}...")

        # Resize frame to target resolution
        frame = cv2.resize(frame, (target_width, target_height))
        annotated_frame = frame.copy()  # Use a copy for annotations

        # --- License Plate Detection and OCR ---
        plate_results = plate_model(frame)[0]
        plate_detections = sv.Detections.from_ultralytics(plate_results)

        if len(plate_detections) > 0:
            logging.info(f"Frame {frame_count}: Detected {len(plate_detections)} license plates")
            for i, (xyxy, confidence, class_id) in enumerate(zip(plate_detections.xyxy, plate_detections.confidence, plate_detections.class_id)):
                class_name = plate_results.names[class_id]
                x1, y1, x2, y2 = map(int, xyxy)
                logging.info(f"  Plate {i+1}: Class={class_name}, Confidence={confidence:.2f}, BBox={xyxy}")

                # Crop the bounding box for OCR
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    logging.warning(f"Empty crop for plate {i+1}")
                    continue

                # Resize crop for better OCR accuracy
                crop = cv2.resize(crop, (110, 70))
                text = perform_ocr(crop)
                if text:
                    text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '').replace('-', ' ')
                    logging.info(f"Detected plate number: {text}")

                    # Draw bounding box and license plate number
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for plates
                    cv2.putText(annotated_frame, text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # No drawing if no OCR text detected
        else:
            logging.info(f"Frame {frame_count}: No license plates detected")

        # --- Helmet Detection ---
        helmet_results = helmet_model(frame)[0]
        helmet_detections = sv.Detections.from_ultralytics(helmet_results)

        if len(helmet_detections) > 0:
            logging.info(f"Frame {frame_count}: Detected {len(helmet_detections)} helmet-related objects")
            for i, (xyxy, confidence, class_id) in enumerate(zip(helmet_detections.xyxy, helmet_detections.confidence, helmet_detections.class_id)):
                class_name = helmet_results.names[class_id]  # 'With Helmet' or 'Without Helmet'
                logging.info(f"  Helmet {i+1}: Class={class_name}, Confidence={confidence:.2f}, BBox={xyxy}")

        # Annotate helmet detections using supervision
        annotated_frame = helmet_box_annotator.annotate(scene=annotated_frame, detections=helmet_detections)
        annotated_frame = helmet_label_annotator.annotate(scene=annotated_frame, detections=helmet_detections)

        # Write and display
        out.write(annotated_frame)
        cv2.imshow("Webcam", annotated_frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Verify output video resolution
    output_cap = cv2.VideoCapture(output_file)
    if output_cap.isOpened():
        out_width = int(output_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_height = int(output_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"Output resolution: {out_width}x{out_height}")
        output_cap.release()
    else:
        logging.error("Could not open output video file to verify resolution.")

    logging.info(f"Output saved as {output_file}")

@app.command()
def webcam(output_file: str = "output1.mp4"):
    typer.echo("Starting video processing...")
    process_webcam(output_file)

if __name__ == "__main__":
    app()