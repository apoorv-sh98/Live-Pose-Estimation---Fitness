import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Read an image from your directory (modify the path as needed)
# image = cv2.imread('path_to_your_image.jpg')

capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv2.CAP_PROP_FPS)


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 
counterl=0
counterr = 0
stagel = ""
stager = ""
# print(f"font:{font} width:{width} height:{height} fps:{fps}")
while capture.isOpened():
    # Capture frame-by-frame
    ret, frame = capture.read()

    if frame is not None:
        # print("frame not none")
        # Process the image for pose detection
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # print("recieved results")

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            shoulderr = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbowr = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wristr = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            # Calculate angle
            anglel = calculate_angle(shoulder, elbow, wrist)
            angler = calculate_angle(shoulderr, elbowr, wristr)
            
            # Visualize angle
            cv2.putText(frame, str(anglel), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            cv2.putText(frame, str(angler), 
                           tuple(np.multiply(elbowr, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            # Curl counter logic
            if anglel > 160:
                stagel = "down"
            if anglel < 30 and stagel =='down':
                stagel="up"
                counterl +=1
                print(f"left: {counterl}")
             # Curl counter logic
            if angler > 160:
                stager = "down"
            if angler < 30 and stager =='down':
                stager="up"
                counterr +=1
                print(f"right: {counterr}")
        except:
            pass
        annotated_image = frame.copy()
        annotated_image2 = frame.copy()
        # Render curl counter
        # Setup status box
        cv2.rectangle(annotated_image, (0,0), (450,210), (245,117,16), -1)
        
        # Rep data
        cv2.putText(annotated_image, 'LEFT', (150, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(annotated_image, 'REPS', (15,110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(annotated_image, str(counterl), 
                    (30,180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # # Stage data
        cv2.putText(annotated_image, 'STAGE', (265,110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(annotated_image, stagel, 
                    (260,180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        cv2.rectangle(annotated_image, (550,0), (1100,210), (245,117,16), -1)
        ## NOW RIGHT
        cv2.putText(annotated_image, 'RIGHT', (750,50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(annotated_image, 'REPS', (615,110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(annotated_image, str(counterr), 
                    (630,180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # # Stage data
        cv2.putText(annotated_image, 'STAGE', (865,110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(annotated_image, stager, 
                    (860,180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        # Draw pose landmarks on the image
        # mp_drawing = mp.solutions.drawing_utils
        
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2))
        # mp_drawing.draw_landmarks(annotated_image2, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2))                
        # for lndmrk in mp_pose.PoseLandmark:
        #     print(lndmrk)
        # Display the image
        cv2.imshow('Pose Detection', annotated_image)
        # cv2.imshow('Pose Detection', annotated_image2)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     

capture.release()
cv2.destroyAllWindows()

