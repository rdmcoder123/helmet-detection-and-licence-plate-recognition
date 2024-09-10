import easyocr
import csv
import os
import numpy as np
import pandas as pd
import cv2
import re
from ultralytics import YOLO
import math
import cvzone
import torch
from sample_img_text import predict_number_plate
from paddleocr import PaddleOCR
from datetime import datetime, timedelta
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import requests
import json


creds_json = {
    "installed": {
        "client_id": "302323034347-fm3h0scq87kves6fk645q29i1ki1n09q.apps.googleusercontent.com",
        "project_id": "helmet-detection-96607",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": "YOUR SECRET",
        "redirect_uris": ["http://localhost"]
    }
}

# Scopes for accessing Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Authenticate Google Drive API
def authenticate_google_drive():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(creds_json, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return creds

# Upload file to Google Drive
def upload_file_to_drive(file_path, folder_id, creds):
    url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
    metadata = {
        "name": os.path.basename(file_path),
        "parents": [folder_id]
    }
    files = {
        "data": ("metadata", json.dumps(metadata), "application/json"),
        "file": open(file_path, "rb")
    }
    headers = {
        "Authorization": f"Bearer {creds.token}"
    }
    response = requests.post(url, headers=headers, files=files)
    if response.status_code == 200:
        file_info = response.json()
        return f"https://drive.google.com/file/d/{file_info['id']}/view?usp=sharing"
    else:
        print("Failed to upload file.")
        print(response.text)
        return None

# Update CSV file with the drive link
def update_csv_file(csv_file, timestamp, vechicle_number, drive_link):
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, vechicle_number, drive_link])

# Initialize the CSV file
csv_file = 'vehicle_data.csv'
csv_header = ['Timestamp', 'Vehicle Number', 'Drive Link']
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

cap = cv2.VideoCapture("C:/Users/Praveen/Downloads/WhatsApp Video 2024-09-08 at 8.10.25 AM.mp4")  # For videos

model = YOLO("runs/detect/train3/weights/best.pt")  # after training update the location of best.pt

device = torch.device("cpu")  # Change to 'cuda' if using GPU

classNames = ["with helmet", "without helmet", "rider", "number plate"]
num = 0
old_npconf = 0

# grab the width, height, and fps of the frames in the video stream.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to download and load model into memory

# Time tracking
last_save_time = datetime.now()
save_interval = timedelta(seconds=2)  # Save data every 2 seconds

last_detected_number = None
last_timestamp = None

while True:
    success, img = cap.read()
    # Check if the frame was read successfully
    if not success:
        break
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(new_img, stream=True, device=device)
    for r in results:
        boxes = r.boxes
        li = dict()
        rider_box = list()
        xy = boxes.xyxy
        confidences = boxes.conf
        classes = boxes.cls
        new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)
        try:
            new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]
            # Get the indices of the rows where the value in column 1 is equal to 5.
            indices = torch.where(new_boxes[:, -1] == 2)
            # Select the rows where the mask is True.
            rows = new_boxes[indices]
            # Add rider details in the list
            for box in rows:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                rider_box.append((x1, y1, x2, y2))
        except:
            pass
        for i, box in enumerate(new_boxes):
            # Bounding box
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box[4] * 100)) / 100
            # Class Name
            cls = int(box[5])
            if classNames[cls] == "without helmet" and conf >= 0.5 or classNames[cls] == "rider" and conf >= 0.45 or \
                    classNames[cls] == "number plate" and conf >= 0.5:
                if classNames[cls] == "rider":
                    rider_box.append((x1, y1, x2, y2))
                if rider_box:
                    for j, rider in enumerate(rider_box):
                        if x1 + 10 >= rider_box[j][0] and y1 + 10 >= rider_box[j][1] and x2 <= rider_box[j][2] and \
                                y2 <= rider_box[j][3]:
                            # highlight or outline objects detected by object detection models
                            cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
                            cvzone.putTextRect(img, f"{classNames[cls].upper()}", (x1 + 10, y1 - 10), scale=1.5,
                                               offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
                            li.setdefault(f"rider{j}", [])
                            li[f"rider{j}"].append(classNames[cls])
                            if classNames[cls] == "number plate":
                                npx, npy, npw, nph, npconf = x1, y1, w, h, conf
                                crop = img[npy:npy + h, npx:npx + w]
                        if li:
                            for key, value in li.items():
                                if key == f"rider{j}":
                                    if len(list(set(li[f"rider{j}"]))) == 3:
                                        try:
                                            vechicle_number, conf = predict_number_plate(crop, ocr)
                                            if vechicle_number and conf:
                                                cvzone.putTextRect(img, f"{vechicle_number} {round(conf*100, 2)}%",
                                                                   (x1, y1 - 50), scale=1.5, offset=10,
                                                                   thickness=2, colorT=(39, 40, 41),
                                                                   colorR=(105, 255, 255))
                                                current_time = datetime.now()
                                                if current_time - last_save_time >= save_interval:
                                                    # Get the current timestamp
                                                    last_timestamp = current_time.strftime('%Y-%m-%d_%H-%M-%S')
                                                    # Write to CSV
                                                    with open(csv_file, 'a', newline='') as file:
                                                        writer = csv.writer(file)
                                                        writer.writerow([last_timestamp, vechicle_number])
                                                    last_save_time = current_time
                                                    last_detected_number = vechicle_number
                                        except Exception as e:
                                            print(e)
        # Display the frame
        output.write(img)
        cv2.imshow('Video', img)
        li = list()
        rider_box = list()

        # Exit the program if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            output.release()
            break

# Release resources
cap.release()
output.release()
cv2.destroyAllWindows()

# After processing the video, upload the file to Google Drive and update CSV file
def main():
    # Authenticate Google Drive
    creds = authenticate_google_drive()

    if last_detected_number and last_timestamp:
        # Generate file name
        file_name = f"{last_detected_number}_{last_timestamp}.mp4"
        
        # Rename your output file
        os.rename('output.mp4', file_name)
        
        # Upload the file to Google Drive
        folder_id = '1i1r5DI14pRPMpN8EdzvDH2KpkY2w2Pah'  # Replace with your folder ID
        drive_link = upload_file_to_drive(file_name, folder_id, creds)
        
        # If upload is successful, update the CSV file
        if drive_link:
            update_csv_file(csv_file, last_timestamp, last_detected_number, drive_link)
            print(f"File uploaded and CSV updated. Drive link: {drive_link}")
        else:
            print("Failed to upload file or update CSV.")
    else:
        print("No license plate detected or timestamp available.")

# Run the main function after video processing
if __name__ == "__main__":
    main()
