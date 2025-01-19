from flask import Flask, request, jsonify
import cv2
import numpy as np
import sys

# Add the path to your Crowd Detection module
sys.path.append(".")  # Assuming Crowd_Detection.py is in the same directory

from Crowd_Detection import detect_crowd  # Import your existing AI model code

app = Flask(__name__)

@app.route('/detect-crowd', methods=['POST'])
def detect_crowd_endpoint():
    try:
        # Retrieve the image from the request
        image_file = request.files['image']
        img_bytes = np.frombuffer(image_file.read(), np.uint8)
        
        # Convert the image to OpenCV format
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        # Run the crowd detection model
        persons_count, alarm_triggered, alarm_message = detect_crowd(img)
        
        # Send the results back to the frontend
        return jsonify({
            "crowdCount": persons_count,
            "alarmTriggered": alarm_triggered,
            "message": alarm_message
        })
    except Exception as e:
        print(f"Error processing the image: {e}")
        return jsonify({"error": "Error processing the image. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True)
