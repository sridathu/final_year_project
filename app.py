from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("malaria_vgg16_model.h5")

def malaria_severity_prediction(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Error loading image. Please check the file path."}

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to detect blood smear region
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return {"error": "No blood smear detected."}

    # Get bounding box of the largest contour (assumed to be blood smear)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Crop the image to the detected blood smear region
    blood_region = image[y:y+h, x:x+w]

    # Convert the cropped image to HSV
    hsv_image = cv2.cvtColor(blood_region, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for infected regions
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])

    # Create mask for infected regions
    mask = cv2.inRange(hsv_image, lower_purple, upper_purple)

    # Corrected total pixel count (excluding black areas)
    total_pixels = blood_region.shape[0] * blood_region.shape[1]
    infected_pixels = np.count_nonzero(mask)  # Only count white pixels in mask
    infection_percentage = (infected_pixels / total_pixels) * 100

    # Ensure percentage is within valid bounds
    infection_percentage = min(max(infection_percentage, 0), 100)

    # Determine severity level
    if infection_percentage == 0:
        severity = "No Infection"
    elif infection_percentage <= 5:
        severity = "Mild"
    elif infection_percentage <= 15:
        severity = "Moderate"
    elif infection_percentage <= 50:
        severity = "Severe"
    else:
        severity = "Critical"

    return {"infection_percentage": round(infection_percentage, 2), "severity": severity}


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = malaria_severity_prediction(filepath)
            return render_template("result.html", result=result, image_filename=filename)
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)