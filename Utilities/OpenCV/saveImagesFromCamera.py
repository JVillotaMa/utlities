import cv2 as cv
import time
import os

# Define the folder where you want to save the images
folder_path = 'path/to/your/folder'

# Check if the folder exists, if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

capture = cv.VideoCapture(0)  # Accept integer (Reference to a webcam) or a path to video file. 

while True:
    isTrue, frame = capture.read()  # isTrue is a boolean, returns if the frame has been read correctly
    if not isTrue:
        print("Failed to capture image.")
        break
    
    cv.imshow('Video', frame)  # Show the frame in window 'Video'
    
    # Check if the 's' key is pressed
    key = cv.waitKey(1) & 0xFF  # Wait for a key press for 1 millisecond
    if key == ord('s'):  # If 'd' is pressed
        # Define the full path to save the image in the desired folder
        image_path = os.path.join(folder_path, f'saved_frame_{int(time.time())}.jpg')# You can also add unique filenames if needed
        cv.imwrite(image_path, frame)  # Save the frame to the specified folder
        print(f"Frame saved as '{image_path}'")

    if key == ord('q'):  # Press 'q' to quit
        break

capture.release()  # Release the capture object
cv.destroyAllWindows()  # Destroy all OpenCV windows
