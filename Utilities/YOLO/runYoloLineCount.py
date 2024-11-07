import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('best.pt')
print(model.names)

# Initialize the webcam
webcamera = cv2.VideoCapture(2)
# webcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# webcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Line coordinates (change these as needed)
line_x = 400  # Y-coordinate of the horizontal line
line_color = (0, 255, 0)  # Green line
line_thickness = 2

# Dictionary to track objects' last positions and whether they've been counted
object_positions = dict()
object_counter = 0

while True:
    success, frame = webcamera.read()
    if not success:
        break
    
    # Get detections from YOLO
    results = model.track(frame, classes=0, conf=0.8, imgsz=480)

    # Draw the counting line
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[1]), line_color, line_thickness)

    # Loop over each detected object
    for box in results[0].boxes:
        # Get bounding box info
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates of the bounding box
        label = box.cls  # Object class label
        obj_id = int(box.id.item())  # Unique ID for each tracked object

        # Calculate the center of the bounding box
        center_x = (x1 + x2) // 2  # Center x-coordinate of the bounding box

        # Check if the object ID has been seen in previous frames
        if obj_id in object_positions:
            prev_center_x = object_positions[obj_id]
            
            # Check if the object moved from the left side of the line to the right
            if prev_center_x < line_x and center_x >= line_x:
                object_counter += 1  # Increment the counter for objects crossing
        
        # Update the object’s current center_x position
        object_positions[obj_id] = center_x
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the count on the frame
    cv2.putText(frame, f"Total Crossed: {object_counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with annotations
    cv2.imshow("Live Camera", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
webcamera.release()
cv2.destroyAllWindows()
