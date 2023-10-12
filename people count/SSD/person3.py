import cv2
import math
# Load the pre-trained MobileNet-SSD model and its configuration
model_path = 'MobileNetSSD_deploy.caffemodel'
config_path = 'MobileNetSSD_deploy.prototxt'

# Load the model and configuration
net = cv2.dnn.readNet(model_path, config_path)

# Define the class labels for COCO dataset
class_labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# Open the video file
video = cv2.VideoCapture('people1.mp4')  # Replace with the path to your video file

# Initialize variables
people_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Prepare the image for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Set the input to the network
    net.setInput(blob)
    detections = net.forward()

    pp=0
    # Loop through the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjust the confidence threshold as needed
            class_id = int(detections[0, 0, i, 1])
            if class_id == class_labels.index('person'):
                people_count += 0.02



    # Display the count on the frame
    cv2.putText(frame, f'People Count: {int(people_count)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
video.release()
cv2.destroyAllWindows()

print(f'Total People Counted: {people_count}')
