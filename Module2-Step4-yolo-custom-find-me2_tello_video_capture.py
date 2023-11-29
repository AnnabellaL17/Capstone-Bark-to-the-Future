import cv2
import torch
import time
from djitellopy import Tello

# Initialize Tello drone
tello = Tello()
tello.connect()

# Paths (change these paths as per your system)

weights_path = "C:\\Users\\psuuvaru\\OneDrive - Cliffwater\\Documents\\yolov5\\exp10-20231126T171648Z-001\\exp10\\weights\\best.pt"

# Setup YOLOv5 with custom model weights

model = torch.hub.load('C:\\Users\\psuuvaru\\OneDrive - Cliffwater\\Documents\\yolov5', 'custom', path=weights_path, source='local')  # 'source' set to 'local' means don't download anything but use local files

## Start video capture from the default computer's camera
cap = cv2.VideoCapture(0)

# Create OpenCV window
#cv2.namedWindow("Tello Video Stream", cv2.WINDOW_NORMAL)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
out = cv2.VideoWriter('tello_video.avi', fourcc, 10.0, (960, 720))  # Adjust the filename and frame rate as needed


#time.sleep(60)

try:
    
    # Start video stream
    tello.streamon()
    #tello.takeoff()

    #tello.move_forward(50)
    #time.sleep(15)

    #tello.move_back(50)


    while True:
        # Get the latest frame from the video stream
        frame = tello.get_frame_read().frame
        # Pass the frame to YOLOv5 for object detection with confidence threshold set to 0.25
        #   results = model(frame_rgb, conf=0.25)  # Set confidence threshold here
        model.conf = 0.50 # Set confidence threshold
        results = model(frame)

        # Display the frame in the OpenCV window
        rendered_frame = results.render()[0]
        cv2.imshow("Tello Video Stream", rendered_frame)
        # Write the frame to the video file
        out.write(frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop video stream, release the video writer, and close OpenCV window
    #tello.land()
    tello.streamoff()
    out.release()
    cv2.destroyAllWindows()
    tello.end()