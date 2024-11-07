import cv2 as cv



capture = cv.VideoCapture(2)  # Accept integer (Reference to a  webcam) or a path to video file. 

while True:
    isTrue, frame = capture.read() # isTrue is a boolean, returnns if frame has been read correctly, and returns the frame
    cv.imshow('Video',frame) # show the frame in window 'Video'
    if cv.waitKey(20) & 0xFF==ord('q'): # 0xFF click key d. Or 20 seconds has been pased
        break;

capture.release() # Release the pointer to the image
cv.destroyAllWindows() # Destroy all windows
