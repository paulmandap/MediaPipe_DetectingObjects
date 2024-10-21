import cv2
import os

outputdir = "newDataset"
os.makedirs(outputdir, exist_ok=True)

cap = cv2.VideoCapture('output.avi')

fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
frame_count = 0
saved_frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save every 2nd frame (or adjust based on fps)
    if frame_count % int(fps / 2) == 0:  # Adjust 2 to desired frame rate
        frame_filename = os.path.join(outputdir, f"frame{saved_frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

    frame_count += 1

cap.release()
print(f"Extracted {saved_frame_count} frames.")