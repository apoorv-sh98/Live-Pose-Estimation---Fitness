import mediapipe as mp
import cv2

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Read an image from your directory (modify the path as needed)
# image = cv2.imread('path_to_your_image.jpg')

capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv2.CAP_PROP_FPS)
print(f"font:{font} width:{width} height:{height} fps:{fps}")
while capture.isOpened():
    # Capture frame-by-frame
    ret, frame = capture.read()

    if frame is not None:
        print("frame not none")
        # Process the image for pose detection
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # print("recieved results")

        # Draw pose landmarks on the image
        mp_drawing = mp.solutions.drawing_utils
        annotated_image = frame.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display the image
        cv2.imshow('Pose Detection', annotated_image)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()