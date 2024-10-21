import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

# Initialize Mediapipe face mesh and hand solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Set up drawing specifications for face mesh
face_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=[4, 244, 4])

# Settings for hand landmarks display
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# Function to draw hand landmarks on the image
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.multi_hand_landmarks
    handedness_list = detection_result.multi_handedness
    annotated_image = np.copy(rgb_image)

    # Loop through detected hands to visualize landmarks
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw hand landmarks
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks.landmark
        ])
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Get the top-left corner of the detected hand's bounding box
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
        y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image
        cv2.putText(annotated_image, f"{handedness.classification[0].label}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Create a face mesh object and hand landmark detector with Mediapipe
with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh, \
    mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()  # Capture frame-by-frame from webcam
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the BGR frame to RGB for Mediapipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find face landmarks
        face_results = face_mesh.process(rgb_frame)

        # Process the frame to find hand landmarks
        hand_results = hands.process(rgb_frame)

        # Check if face landmarks were detected
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw the face landmarks on the frame
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=face_drawing_spec,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # Check if hand landmarks were detected and draw them
        if hand_results.multi_hand_landmarks:
            frame = draw_landmarks_on_image(frame, hand_results)

        # Display the resulting frame
        cv2.imshow('Face and Hand Mesh Live', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
