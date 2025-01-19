import cv2
import numpy as np
from twilio.rest import Client
import pygame

# Twilio configuration for sending SMS alerts
account_sid = 'AC72412f3e10d993749cd4c8fed09025e4'  # Replace with your Twilio Account SID
auth_token = 'e2375855e1a98d245a7cf40b5447d765'     # Replace with your Twilio Auth Token
twilio_phone_number = '+14153906775'                # Replace with your Twilio phone number
your_phone_number = '+917701847323'                 # Replace with the number you want to receive SMS alerts              

# Initialize Twilio client
client = Client(account_sid, auth_token)

# Initialize pygame for playing alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(r"Hackathons/2. Rajasthan Police Hackathon/Codes/Crowd/Crowd Audio.mp3")  # Path to your alarm audio file

# Load YOLO model and configuration
yolo_weights = r"Hackathons/2. Rajasthan Police Hackathon/Codes/Crowd/Crowd_yolov3.weights"
yolo_cfg = r"Hackathons/2. Rajasthan Police Hackathon/Codes/Crowd/Crowd_yolov3.cfg"
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
layer_names = net.getUnconnectedOutLayersNames()

def detect_crowd(image):    
    # Set the threshold for crowd detection (e.g., max number of people before alerting)
    crowd_threshold = 5
    alarm_triggered = False
    alarm_message = "Crowd is within limits."
    
    # Pre-process the input image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Detect only 'person' objects (YOLO class ID 0 for person)
            if class_id == 0 and confidence > 0.2:
                # Object detection properties (bounding box)
                center_x, center_y, width, height = (obj[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype("int")
                x, y = int(center_x - width / 2), int(center_y - height / 2)

                boxes.append([x, y, x + width, y + height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_boxes = [boxes[i] for i in indices]

    # Count the number of persons detected in the image
    persons_count = len(detected_boxes)

    # If the count exceeds the threshold, trigger alarm and send SMS
    if persons_count > crowd_threshold:
        alarm_message = "Crowd limit exceeded! Take necessary actions."
        alarm_triggered = True

        # Send SMS alert using Twilio
        client.messages.create(
            body=alarm_message,
            from_=twilio_phone_number,
            to=your_phone_number
        )

        # Play alarm sound
        pygame.mixer.Sound.play(alarm_sound)

    return persons_count, alarm_triggered, alarm_message
