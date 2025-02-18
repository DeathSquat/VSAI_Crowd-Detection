import cv2
import numpy as np
from twilio.rest import Client
import pygame
import time  

# Load YOLO model and configuration
yolo_weights = "1. AI Camera/Crowd/Crowd_yolov3.weights"
yolo_cfg = "1. AI Camera/Crowd/Crowd_yolov3.cfg"

# Check file availability and paths
try:
    with open(yolo_weights, 'rb') as f:
        pass
except FileNotFoundError:
    print(f"Error: '{yolo_weights}' not found.")
    exit(1)

try:
    with open(yolo_cfg, 'rb') as f:
        pass
except FileNotFoundError:
    print(f"Error: '{yolo_cfg}' not found.")
    exit(1)

net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# Check if the network was loaded successfully
if net.empty():
    print("Error: Failed to load YOLO network.")
    print("Please ensure that the YOLO configuration and weights files are correct and compatible with OpenCV.")
    print("You may also try updating OpenCV to the latest version for better compatibility.")
    exit(1)
else:
    print("YOLO network loaded successfully.")

layer_names = net.getUnconnectedOutLayersNames()

# Initialize variables
crowd_threshold = 0
alarm_message = "Crowd is within limits."
alarm_triggered = False

# Twilio configuration
account_sid = 'Your_Account_ID'
auth_token = 'Your_Auth_Token'
twilio_phone_number = 'Your_Twilio_PN'
your_phone_number = 'Your_PN'

# Initialize Twilio client
client = Client(account_sid, auth_token)

# Initialize pygame for sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("1. AI Camera/Crowd/Crowd Audio.mp3")

# Open video capture
cap = cv2.VideoCapture(0)  

# Function to make phone call with a short delay
def make_phone_call_with_delay():
    time.sleep(5) 
    call = client.calls.create(
        to=your_phone_number,
        from_=twilio_phone_number,
        url="http://demo.twilio.com/docs/voice.xml"  
    )
    print(f"Phone call initiated. SID: {call.sid}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(layer_names)

    # Apply Non-Maximum Suppression (NMS)
    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == 0 and confidence > 0.2:  # Detect person (class_id=0)
                center_x, center_y, width, height = (obj[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype("int")
                x, y = int(center_x - width / 2), int(center_y - height / 2)

                boxes.append([x, y, x + width, y + height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Loop through the remaining boxes after NMS
    detected_boxes = [boxes[i] for i in indices]

    # Count persons in the frame
    persons_count = len(detected_boxes)

    # Add a semi-transparent overlay
    overlay = frame.copy()
    alpha = 0.6
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)  # Black semi-transparent overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Display the crowd count and alarm message
    text_color = (0, 255, 0) if not alarm_triggered else (0, 0, 255)
    cv2.putText(frame, f"Number of Persons: {persons_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    cv2.putText(frame, alarm_message, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # Draw bounding boxes
    for box in detected_boxes:
        x, y, x_end, y_end = box
        color = (0, 255, 0) if persons_count <= crowd_threshold else (0, 0, 255)  # Green or Red based on threshold
        cv2.rectangle(frame, (x, y), (x_end, y_end), color, 2)

    # Check if crowd count exceeds the threshold
    if persons_count > crowd_threshold and not alarm_triggered:
        alarm_triggered = True
        alarm_message = "Crowd limit exceeded! Take necessary actions."

        # Send SMS using Twilio
        message = client.messages.create(
            body=alarm_message,
            from_=twilio_phone_number,
            to=your_phone_number
        )

        # Initiate a phone call with a short delay
        make_phone_call_with_delay()

        # Play alarming sound
        pygame.mixer.Sound.play(alarm_sound)

    elif persons_count <= crowd_threshold and alarm_triggered:
        alarm_triggered = False
        alarm_message = "Crowd is within limits."

    # Show the frame
    cv2.imshow("Crowd Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
