import cv2
from ultralytics import YOLO
import time

# Function to load the YOLO model
def load_model(model_path='best.pt'):
    model = YOLO(model_path)
    print("Available classes:", model.names)
    return model

# Function to initialize the webcam
def initialize_camera(camera_index=2):
    webcamera = cv2.VideoCapture(camera_index)
    if not webcamera.isOpened():
        print("Error: Could not open camera.")
        return None
    return webcamera

# Function to draw a counting line on the frame
def draw_counting_line(frame, line_x, color=(0, 255, 0), thickness=2):
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), color, thickness)

# Function to process detections and track object positions
def process_detections(frame, results, object_positions, object_counts, line_x, model):
    for box in results[0].boxes:
        # Extract bounding box and object information
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = int(box.cls.item())  # Object class label as an integer
        obj_id = int(box.id.item())  # Unique object ID assigned by tracker
        center_x = (x1 + x2) // 2    # Calculate the center x-coordinate of the bounding box

        # Check if object has crossed the line
        if obj_id in object_positions:
            prev_center_x = object_positions[obj_id]
            if prev_center_x < line_x and center_x >= line_x:
                # Object has crossed from left to right
                class_name = model.names[label]
                
                # Update count for the class of object that crossed the line
                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1
                
                # Print crossing event
                print(f"Object ID {obj_id} (class {class_name}) crossed the line.")

        # Update the current x position for the object
        object_positions[obj_id] = center_x

        # Draw bounding box and label
        draw_bounding_box(frame, x1, y1, x2, y2, label, obj_id, model)

# Function to draw bounding boxes and labels on the frame
def draw_bounding_box(frame, x1, y1, x2, y2, label, obj_id, model):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, f"ID: {obj_id} Class: {model.names[label]}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Main function to run the object tracking and counting
def main():
    model = load_model('best.pt')
    webcamera = initialize_camera(2)
    if not webcamera:
        return
    
    line_x = 500  # Set the x-coordinate of the vertical counting line
    object_positions = {}  # Track the last x position of each object by ID
    object_counts = {}  # Count objects that have crossed the line by class

    while True:
        success, frame = webcamera.read()
        if not success:
            break

        # Get detections from YOLO
        results = model.track(frame, classes=None, conf=0.8, imgsz=480)
        draw_counting_line(frame, line_x)

        # Process detections and update counts
        process_detections(frame, results, object_positions, object_counts, line_x, model)

        # Display the counts on the frame
        y_offset = 50
        for class_name, count in object_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset += 30

        # Show the frame with annotations
        cv2.imshow("Live Camera", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release resources
    webcamera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()