from imutils.video import VideoStream
import argparse
import imutils
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help="path to where the face cascade resides")
ap.add_argument("-o", "--output", required=True, help="path to output directory")
args = vars(ap.parse_args())

# Initialize the OpenCV Haar cascade for face detection from disk
detector = cv2.CascadeClassifier(args["cascade"])

# Initialize the video stream
vs = VideoStream(src=0).start()
total = 0

# Get user input for name or photo choice
choice = int(input("Press 1 for Name, Press 2 for Photo : "))
if choice == 1:
    name = input("Enter name: ")

else:
    name = "Unknown"


print("[INFO] starting video stream...")

# ...

while True:
    try:
        frame = vs.read()
        orig = frame.copy()
        frame = imutils.resize(frame, width=400)

        # Detect faces in the grayscale frame
        rects = detector.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30))

        # Loop over the face detections and draw rectangles on the frame
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # If the 'k' key was pressed, save the face image with a unique name
        if key == ord("k"):
            filename = "{}.png".format(name)
            p = os.path.sep.join([args["output"], filename])
            cv2.imwrite(p, orig)
            total += 1

        # If the 'q' key was pressed, break from the loop
        elif key == ord("q"):
            break  # Break out of the loop if 'q' is pressed

    except KeyboardInterrupt:
        break  # Handle KeyboardInterrupt to exit the loop gracefully

# Print the total faces saved and do a bit of cleanup
print("[INFO] {} face images stored for {}".format(total, name))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
