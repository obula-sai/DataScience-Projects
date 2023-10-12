import cv2
import numpy as np

# Load YOLO model and configure its parameters
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Replace with the correct file paths
layer_names = net.getLayerNames()
output_layers = [layer for layer in layer_names if 'Detection' in net.getLayer(layer).type]

# Load video file
video = cv2.VideoCapture("people.mp4")  # Replace with the correct video file path

# Initialize variables
people_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects in the frame using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []

    # Process the YOLO output to count people
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0:  # Class 0 represents 'person' in YOLO
                people_count += 0.02

    # Display the count on the frame
    cv2.putText(frame, f'People Count: {int(people_count)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Video", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
video.release()
cv2.destroyAllWindows()

print(f"Total People Counted: {people_count}")


