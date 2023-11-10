import cv2
import time
import os
from pathlib import Path


def get_download_path():
    """Returns the default downloads path for Linux, Mac and Windows."""
    if os.name == 'nt':  # for Windows
        import winreg
        sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
        downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
            location = winreg.QueryValueEx(key, downloads_guid)[0]
        return location
    elif os.name == 'posix':  # for macOS and Linux
        return str(Path.home() / 'Downloads')
    else:
        raise ValueError("Unsupported operating system")

# Use the get_download_path function to set the directory path
directory_name = 'SecureCam'
downloads_path = get_download_path()
secure_cam_directory_path = os.path.join(downloads_path, directory_name)
directory_paths = [secure_cam_directory_path]

# Create the directory if it does not exist
os.makedirs(secure_cam_directory_path, exist_ok=True)

# Initialize the face and body detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set the resolution: 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip frame horizontally to correct mirrored image
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face and Body Detection', frame)

    if (len(faces) > 0 or len(bodies) > 0) and time.time() - last_time >= 5:
        last_time = time.time()
        for path in directory_paths:
            img_name = os.path.join(path, f"SecureCam_{int(last_time)}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Image {img_name} saved.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
