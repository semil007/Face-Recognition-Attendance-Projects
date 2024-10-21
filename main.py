import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import time
import winsound

# Initialize a dictionary to track names and their attendance status
attendance_history = {}

# Function to show a success message
def show_success_message(name, action):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showinfo("Attendance Marked", f"{action} for {name}!")
    root.destroy()

# Function to play sound alert
def play_sound():
    winsound.Beep(1000, 500)  # Beep sound for 500 ms

# Load training images and names
path = r'D:\Face-Recognition-Attendance-Projects-main\Face-Recognition-Attendance-Projects-main\Training_images'
images = []
classNames = os.listdir(path)

# Display student names before encoding
print("Encoding the following student images:")
for cl in classNames:
    print(cl)
    curImg = cv2.imread(os.path.join(path, cl))
    images.append(curImg)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name, action):
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    dateString = now.strftime('%Y-%m-%d')

    # Load or create the attendance DataFrame
    try:
        df = pd.read_csv(r'D:\Face-Recognition-Attendance-Projects-main\Face-Recognition-Attendance-Projects-main\Attendance.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Name', 'Date', 'IN', 'OUT'])

    # Create a new entry for the student
    if action == "IN":
        new_entry = pd.DataFrame({'Name': [name], 'Date': [dateString], 'IN': [dtString], 'OUT': ['']})
    else:  # action == "OUT"
        new_entry = pd.DataFrame({'Name': [name], 'Date': [dateString], 'IN': [''], 'OUT': [dtString]})

    df = pd.concat([df, new_entry], ignore_index=True)

    # Save the updated DataFrame to CSV
    df.to_csv(r'D:\Face-Recognition-Attendance-Projects-main\Face-Recognition-Attendance-Projects-main\Attendance.csv', index=False)
    
    # Show success message
    show_success_message(name, action)
    play_sound()  # Play sound alert

# Encode known faces
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Load existing attendance data to set initial states
try:
    df = pd.read_csv(r'D:\Face-Recognition-Attendance-Projects-main\Face-Recognition-Attendance-Projects-main\Attendance.csv')
    for index, row in df.iterrows():
        if row['IN'] and not row['OUT']:
            attendance_history[row['Name']] = {'status': 'IN', 'time': time.time()}  # Store time as well
except FileNotFoundError:
    attendance_history = {}

cap = cv2.VideoCapture(0)

def close_camera():
    cap.release()
    cv2.destroyAllWindows()
    exit()  # Ensure the program exits completely

# Main loop
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgS = cv2.resize(img, (320, 240))  # Reduce resolution for faster processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    found_any_student = False  # Flag to check if a registered student is found

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()  # Get student name
            found_any_student = True  # Found a registered student
            y1, x2, y2, x1 = [v * 4 for v in faceLoc]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle around face
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)  # Background for name
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # Put name text

            # Mark attendance with "IN" or "OUT"
            if name not in attendance_history:
                print(f"{name} marked IN")
                markAttendance(name, "IN")
                attendance_history[name] = {'status': 'IN', 'time': time.time()}  # Store status and time
            else:
                # If the student is already marked "IN", check time to mark "OUT"
                if attendance_history[name]['status'] == 'IN':
                    current_time = time.time()
                    if current_time - attendance_history[name]['time'] > 5:  # 5 seconds to mark OUT
                        print(f"{name} marked OUT")
                        markAttendance(name, "OUT")
                        del attendance_history[name]  # Remove from attendance history
                # If the student is marked "OUT", check time to mark "IN"
                elif attendance_history[name]['status'] == 'OUT':
                    current_time = time.time()
                    if current_time - attendance_history[name]['time'] > 5:  # 5 seconds to mark IN
                        print(f"{name} marked IN again")
                        markAttendance(name, "IN")
                        attendance_history[name] = {'status': 'IN', 'time': current_time}  # Reset the time for "IN"

    if not found_any_student:
        cv2.putText(img, "You are not a student", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)  # Show message for non-student

    cv2.imshow('Webcam', img)

    # Exit on window close or 'q' key press
    if cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) < 1 or cv2.waitKey(1) & 0xFF == ord('q'):
        close_camera()
