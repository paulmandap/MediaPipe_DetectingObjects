import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('C:/Paul/VSPython/detect/detect/train4/weights/best.pt')

# Open a connection to the webcam (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object to save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # 20 fps and 640x480 resolution

while True:
    ret, frame = cap.read()  # Capture each frame from the webcam
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, conf=0.5)  # Adjust confidence threshold

    # Check if results are obtained
    for result in results:
        print(result)  # Print result details to debug

        # Annotate the frame
        annotated_frame = result.plot()  # Use plot() for visualization

        # Write the frame to the video file
        out.write(annotated_frame)

        # Display the frame with detections
        cv2.imshow('Frame', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam, video writer, and close any OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()
