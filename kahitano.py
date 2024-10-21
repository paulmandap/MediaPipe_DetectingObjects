import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # 20 fps and 640x480 resolution

while True:
    ret, frame = cap.read()
    if ret:
        # Write the frame to the output video file
        out.write(frame)

        cv2.imshow('Recording', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()