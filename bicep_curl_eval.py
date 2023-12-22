import mediapipe as mp
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from deep_pose_models import SimplePoseModel
from super_gradients.training import models
from super_gradients.common.object_names import Models
from constants import YOLO_pose_dict as pose_dict

# def update(i):
#     ret, frame = cap.read()
#     if ret:
#         # Process the frame for pose estimation
#         annotated_image = SimplePoseModel.run_simple_pose(frame, annotated_image, detector, pose_net, mp_pose)
#         if annotated_image is not None:
#             # Assuming 'annotated_image' is a numpy array in the correct format
#             ax.clear()
#             ax.imshow(annotated_image)
#             ax.axis('off')  # Hide the axis

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

if __name__ == "__main__":
    fig, ax = plt.subplots()
    parser = argparse.ArgumentParser(
        description="Running pose estimation using various models"
    )
    parser.add_argument(
        "-m", 
        "--model", 
        help="Enter the value of model to be run",
        choices=["mp", "sp", "ap"],
        default="sp"
    )
    args = parser.parse_args()
    capture = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(f"font:{font} width:{width} height:{height} fps:{fps}")
    mp_pose = mp.solutions.pose
    typeModel = args.model
    if typeModel == "mp":
        pose = mp_pose.Pose()
    elif typeModel == "sp":
        # detector, pose_net = SimplePoseModel.instantiate_simple_pose()
        yolo_nas_pose = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
    counterl = 0
    counterr = 0
    stagel = ""
    stager = ""
    while capture.isOpened():
        ret, frame = capture.read()
        if frame is not None:
            annotated_image = frame.copy()
            # calling the model and getting results
            if typeModel == "mp":
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # print(results.pose_landmarks)
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
                    # cv2.putText(frame, str(anglel), 
                    #             tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    #                     )
                    
                    # cv2.putText(frame, str(angler), 
                    #             tuple(np.multiply(elbowr, [640, 480]).astype(int)), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    #                     )
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
                mp_drawing = mp.solutions.drawing_utils
                
                mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2))
                # mp_drawing = mp.solutions.drawing_utils
                # mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # Display the image
            elif typeModel == "sp":
                # ani = FuncAnimation(fig, update, interval=50)
                preds = yolo_nas_pose.predict(annotated_image, conf=0.55)
                prediction = preds[0].prediction
                poses = prediction.poses.tolist()
                edge_links = prediction.edge_links.tolist()
                if len(poses) == 0:
                    continue
                for j, coord in enumerate(poses[0]):
                    # if confidence[i][j].asscalar() < 0.2:  # Adjust this threshold as needed
                    #     continue
                    x, y, conf = int(coord[0]), int(coord[1]), coord[2]
                    cv2.circle(annotated_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
                for i, link in enumerate(edge_links):
                    first = link[0]
                    second = link[1]
                    x1 = int(poses[0][first][0])
                    y1 = int(poses[0][first][1])
                    x2 = int(poses[0][second][0])
                    y2 = int(poses[0][second][1])
                    # print(f"{x1} {y1} {x2} {y2}")
                    # cv2.putText(annotated_image, str(first), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1, cv2.LINE_AA)
                    # cv2.putText(annotated_image, str(second), (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1, cv2.LINE_AA)
                    cv2.line(annotated_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
                
                # Get coordinates
                shoulder = [poses[0][pose_dict["LEFT_SHOULDER"]][0],poses[0][pose_dict["LEFT_SHOULDER"]][1]]
                elbow = [poses[0][pose_dict["LEFT_ELBOW"]][0],poses[0][pose_dict["LEFT_ELBOW"]][1]]
                wrist = [poses[0][pose_dict["LEFT_WRIST"]][0],poses[0][pose_dict["LEFT_WRIST"]][1]]
                '''
                shoulderr = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbowr = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wristr = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                '''
                # Calculate angle
                anglel = calculate_angle(shoulder, elbow, wrist)
                print(f"    {anglel}")
                # angler = calculate_angle(shoulderr, elbowr, wristr)
                
                # Visualize angle
                cv2.putText(annotated_image, str(anglel), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                    
                    # cv2.putText(frame, str(angler), 
                    #             tuple(np.multiply(elbowr, [640, 480]).astype(int)), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    #                     )
                    # Curl counter logic
                if anglel > 160:
                    stagel = "down"
                if anglel < 30 and stagel =='down':
                    stagel="up"
                    counterl +=1
                    print(f"left: {counterl}")
                    # Curl counter logic
                    # if angler > 160:
                    #     stager = "down"
                    # if angler < 30 and stager =='down':
                    #     stager="up"
                    #     counterr +=1
                    #     print(f"right: {counterr}")
                # except:
                #     pass
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

                # cv2.rectangle(annotated_image, (550,0), (1100,210), (245,117,16), -1)
                # ## NOW RIGHT
                # cv2.putText(annotated_image, 'RIGHT', (750,50), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1, cv2.LINE_AA)
                # cv2.putText(annotated_image, 'REPS', (615,110), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1, cv2.LINE_AA)
                # cv2.putText(annotated_image, str(counterr), 
                #             (630,180), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # # # Stage data
                # cv2.putText(annotated_image, 'STAGE', (865,110), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1, cv2.LINE_AA)
                # cv2.putText(annotated_image, stager, 
                #             (860,180), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                # # Draw pose landmarks on the image
                # # mp_drawing = mp.solutions.drawing_utils
                
                # mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                #                         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2))
                
            #     # preds.show()
            # else:
            #     print("default")
            
            cv2.imshow('Pose Detection', annotated_image)
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    capture.release()
    cv2.destroyAllWindows()
