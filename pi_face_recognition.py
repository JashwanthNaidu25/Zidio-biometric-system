from imutils.video import VideoStream, FPS
import face_recognition
import argparse
import imutils
import pickle
import cv2
import time
import mysql.connector
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import os

# Global variables and settings
employee_id_counter = 0
recognized_persons = {}  # Stores recognized persons with the last recognition timestamp
cooldown_period = 20  # Cooldown period in seconds

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '19092003',
    'database': 'face_biometric',
}

# Initialize database connection pool
try:
    conn_pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name="mypool", pool_size=5, **db_config
    )
    print("[INFO] Database connection pool created.")
except mysql.connector.Error as err:
    print(f"[ERROR] Database connection error: {err}")
    exit(1)


def process_frame(frame, data, cursor):
    """Processes a single video frame for face recognition and attendance logging."""
    global recognized_persons, employee_id_counter

    frame_resized = imutils.resize(frame, width=500)
    face_locations = face_recognition.face_locations(frame_resized)
    encodings = face_recognition.face_encodings(frame_resized, face_locations)

    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = None

        if True in matches:
            matchedIdx = matches.index(True)
            name = data["names"][matchedIdx]

            current_time = time.time()
            if name not in recognized_persons or (current_time - recognized_persons.get(name, 0)) >= cooldown_period:
                recognized_persons[name] = current_time
                employee_id_counter += 1
                employee_id = employee_id_counter

                try:
                    cursor.execute(
                        "INSERT INTO attendance (employee_id, name, date, time) VALUES (%s, %s, CURDATE(), CURTIME())",
                        (employee_id, name),
                    )
                    print(f"[INFO] Attendance logged for {name} with ID {employee_id}.")
                except mysql.connector.Error as err:
                    print(f"[ERROR] Failed to insert attendance: {err}")

        if name:
            names.append(name)

    for ((top, right, bottom, left), name) in zip(face_locations, names):
        cv2.rectangle(frame_resized, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame_resized, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return frame_resized


def list_testing_images(directory):
    """Lists all .png images in the 'testing' directory."""
    if not os.path.exists(directory):
        print(f"[ERROR] The directory {directory} does not exist.")
        return []

    images = [f for f in os.listdir(directory) if f.lower().endswith('.png')]
    if not images:
        print("[INFO] No .png images found in the directory.")
    else:
        print("[INFO] Found the following .png images:")
        for image in images:
            print(image)
    return images


# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

# Load face encodings
try:
    print("[INFO] Loading encodings...")
    with open(args["encodings"], "rb") as f:
        data = pickle.load(f)
except FileNotFoundError:
    print("[ERROR] Encodings file not found.")
    exit(1)

# List .png images in the 'testing' folder
testing_dir = "C:/Users/Kushal S/Desktop/internship projects/project 11/biometric fb/captured_faces/dataset/testing"
testing_images = list_testing_images(testing_dir)

# Initialize video stream
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Video processing loop
try:
    with conn_pool.get_connection() as conn:
        cursor = conn.cursor()
        with ThreadPoolExecutor() as executor:
            while True:
                frame = vs.read()
                future = executor.submit(process_frame, frame, data, cursor)
                frame_processed = future.result()

                cv2.imshow("Frame", frame_processed)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                fps.update()
except KeyboardInterrupt:
    print("\n[INFO] Exiting on keyboard interrupt.")
except mysql.connector.Error as err:
    print(f"[ERROR] Database error during execution: {err}")
finally:
    fps.stop()
    print(f"[INFO] Elapsed time: {fps.elapsed():.2f}")
    print(f"[INFO] Approx. FPS: {fps.fps():.2f}")

    cursor.close()
    vs.stop()
    cv2.destroyAllWindows()
