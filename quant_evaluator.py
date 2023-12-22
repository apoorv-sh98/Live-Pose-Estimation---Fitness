from super_gradients.training import models
import mediapipe as mp
import os
import cv2
import json
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import argparse
from constants import YOLO_pose_dict, TRUE_point_dict, MEDIAPIPE_pose_dict

def euclidean_distance(true_coords, pred_coords):
    distances = np.sqrt(np.sum((true_coords - pred_coords)**2, axis=1))
    return distances

def mean_squared_error_metric(true_coords, pred_coords):
    mse = mean_squared_error(true_coords, pred_coords)
    return mse

def percentage_of_correct_keypoints(true_coords, pred_coords, distances, threshold=5.0):
    correct_keypoints = np.sum(distances < threshold)
    total_keypoints = len(true_coords)
    pck = correct_keypoints / total_keypoints
    return pck

def intersection_over_union(true_coords, pred_coords, distances, threshold=5.0):
    true_mask = np.zeros_like(true_coords)
    pred_mask = np.zeros_like(pred_coords)

    true_mask[true_coords < threshold] = 1
    pred_mask[pred_coords < threshold] = 1

    intersection = np.sum(np.logical_and(true_mask, pred_mask))
    union = np.sum(np.logical_or(true_mask, pred_mask))
    print(intersection)
    print(union)

    iou = intersection / union
    return iou

def evaluate_metrics(true_list, pred_list, threshold=10.0):
    true_keypoints = np.array(true_list)  # Replace with your true keypoint coordinates
    pred_keypoints = np.array(pred_list)  # Replace with your predicted keypoint coordinates

    # Calculate metrics
    distances = euclidean_distance(true_keypoints, pred_keypoints)
    mse = mean_squared_error_metric(true_keypoints, pred_keypoints)
    pck = percentage_of_correct_keypoints(true_keypoints, pred_keypoints, distances, threshold)
    # iou = intersection_over_union(true_keypoints, pred_keypoints, distances, threshold=10.0)
    dist = [round(distance, 5) for distance in distances]
    
    # Print or use the metrics as needed
    print(f"\n\tEuclidean Distance: {dist}")
    print(f"\tMean Squared Error: {mse}")
    print(f"\tPCK: {pck}")
    # print(f"\tIoU: {iou}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t",
                        "--type",
                        choices=["mp", "yn"],
                        help="Choose the model that is to be run")

    args = parser.parse_args()
    if args.type == "yn":
        yolo_nas_pose = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
    else:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

    annotation_dir = "mpii-annotations-json/"
    
    names = os.listdir(annotation_dir)
    for name in names:
        annotated_image_name = name
        file_name = f"images/{annotated_image_name.split('.')[0]}.jpg"
        print(file_name)
        annotated_image = cv2.imread(file_name)
        image_height, image_width, c = annotated_image.shape
        copyimage = annotated_image.copy()
        jsondict = {}
        with open(os.path.join(dir, annotated_image_name), encoding="utf8", errors='ignore') as f:
            jsondict = json.loads(f.read())
        # print(jsondict)
        image_dict = {}
        for i, id in enumerate(jsondict["j_ids"]):
            x = jsondict['ax'][i]
            y = jsondict['ay'][i]
            image_dict[id] = [x, y]
        print(image_dict)
        for id, coord in image_dict.items():
            # if confidence[i][j].asscalar() < 0.2:  # Adjust this threshold as needed
            #     continue
            print(type(coord[0]))
            x, y = int(coord[0]), int(coord[1])
            print(f'{x} {y}')
            cv2.circle(annotated_image, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
        # cv2.imshow('Pose Detection', annotated_image)
        # cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        common_j = []
        true_j = []
        pred_j = []

        if args.type == "yn":
            # yolo-nas pred
            preds = yolo_nas_pose.predict(annotated_image, conf=0.55)
            prediction = preds[0].prediction
            poses = prediction.poses.tolist()
            edge_links = prediction.edge_links.tolist()
            print(poses)
            print(len(poses))
            # yolonas lists
            for key, value in image_dict.items():
                body_part = TRUE_point_dict[key]
                if body_part not in YOLO_pose_dict.keys():
                    continue
                common_j.append(body_part)
                true_j.append([value[0], value[1]])
                idx = YOLO_pose_dict[body_part]
                x, y = int(poses[0][idx][0]), int(poses[0][idx][1])
                pred_j.append([x, y])
        else:
            #media pipe pred
            results = pose.process(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            landmarks = results.pose_landmarks.landmark
            print(f"pose landmarks: {mp_pose.PoseLandmark}")
            print(f"landmarks: {len(landmarks)}")
            # mediapipe lists
            for key, value in image_dict.items():
                body_part = TRUE_point_dict[key]
                if body_part not in MEDIAPIPE_pose_dict.keys():
                    continue
                common_j.append(body_part)
                true_j.append([value[0], value[1]])
                idx = MEDIAPIPE_pose_dict[body_part]
                x, y = landmarks[idx].x, landmarks[idx].y
                pred_j.append([x, y])
            # normalization for mediapipe
            pred_l = []
            for x, y in pred_j:
                x_px = min(math.floor(x * image_width), image_width - 1)
                y_px = min(math.floor(y * image_height), image_height - 1)
                pred_l.append([x_px, y_px])
            print(pred_l)
        
        print("********* Final Lists for image *********")
        print(common_j)
        print(true_j)
        print(pred_j)
        left_hip = []
        right_hip = []
        for i, joint in enumerate(common_j):
            if joint == "RIGHT_HIP":
                right_hip.append(true_j[i])
            if joint == "LEFT_HIP":
                left_hip.append(true_j[i])
        threshold = euclidean_distance(np.array(right_hip), np.array(left_hip))
        print(f"torso length: {threshold}")
        pred = []
        if args.type == "mp":
            pred = pred_l
        else:
            pred = pred_j
        evaluate_metrics(true_j, pred, 0.20*threshold)

        # to display image
        if args.type == "mp":
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2))
        else:
            if len(poses) == 0:
                continue
            # c = 0
            for j, coord in enumerate(poses[0]):
                # if confidence[i][j].asscalar() < 0.2:  # Adjust this threshold as needed
                #     continue
                x, y, conf = int(coord[0]), int(coord[1]), coord[2]
                # cv2.putText(annotated_image, f"{c}",(x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),1,cv2.LINE_AA)
                cv2.circle(annotated_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
                # c += 1
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
        
            cv2.imshow('Pose Detection', annotated_image)
            cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        

