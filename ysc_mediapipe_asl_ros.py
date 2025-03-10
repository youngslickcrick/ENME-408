#! /usr/bin/env python3
# Chris Iheanacho @ UMBC
# Description: Compares live hand gestures using Sawyer's head camera against saved gestures
# and combines them into a word.
# ysc_mediapipe_asl_ros.py


import rospy
import cv2
import os
import json
import time
import numpy as np
import mediapipe as mp
import intera_interface
from collections import deque
from cv_bridge import CvBridge, CvBridgeError
import math

bridge = CvBridge()
latest_frame = None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, 
                       max_num_hands=1,
                       min_detection_confidence=0.5, 
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


SAVED_GESTURES_PATH = "/home/ysc/ros_ws/src/intera_sdk/intera_examples/scripts/captured_gestures_ros"
saved_landmarks = {}
gesture_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "finish", "backspace"]  # Added 'backspace' to the list

for label in gesture_labels:
    json_path = os.path.join(SAVED_GESTURES_PATH, f"{label}_asl_landmarks_ros.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            saved_landmarks[label] = normalize_landmarks(json.load(f))
            

def camera_callback(img_data, camera_name):
    """Callback function to process images from Sawyer's head camera."""
    global latest_frame
    try:
        frame = bridge.imgmsg_to_cv2(img_data, "bgr8")

        latest_frame = cv2.flip(frame, 1)

    except CvBridgeError as e:
        rospy.logerr(f"CV Bridge Error: {e}")

def normalize_landmarks(landmarks): 
    wrist = landmarks[0]  
    normalized_landmarks = []
    
    for lm in landmarks:
        norm_x = lm['x'] - wrist['x']
        norm_y = lm['y'] - wrist['y']
        norm_z = lm['z'] - wrist['z']
        normalized_landmarks.append({'x': norm_x, 'y': norm_y, 'z': norm_z})

    return normalized_landmarks

def weighted_distance(saved, detected):

    weight_x = 1.0  
    weight_y = 1.0  
    weight_z = 0.5 

    return np.sqrt(
        (weight_x * (saved['x'] - detected['x']))**2 +
        (weight_y * (saved['y'] - detected['y']))**2 +
        (weight_z * (saved['z'] - detected['z']))**2)

def compare_gesture_live(threshold=0.05, hold_time=.5, history_frames=5):

    rospy.init_node("sawyer_gesture_recognition", anonymous=True)


    cameras = intera_interface.Cameras()
    camera_name = "head_camera"

    if not cameras.verify_camera_exists(camera_name):
        rospy.logerr(f"Error: Camera '{camera_name}' not detected. Exiting.")
        return

    rospy.loginfo(f"Starting stream from {camera_name}...")
    cameras.start_streaming(camera_name)

    cameras.set_gain(camera_name, -1)
    cameras.set_exposure(camera_name, -1)


    cameras.set_callback(camera_name, camera_callback, rectify_image=True, callback_args=(camera_name,))

    print("Press 'ESC' to exit the application.")

    gesture_start_time = None
    detected_gesture = None
    confirmed_gesture = None
    recent_detections = deque(maxlen=history_frames) 
    word_spelled = []

    while not rospy.is_shutdown():
        if latest_frame is None:
            continue

        frame = latest_frame.copy()


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        gesture_matched = False 

        height, width = frame.shape[:2]

        box_size = int(height * 0.2)


        top_left = (width // 2 - box_size // 2, int(height * 0.25)) 
        bottom_right = (top_left[0] + box_size, top_left[1] + box_size)

        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))


                detected_landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
                detected_landmarks = normalize_landmarks(detected_landmarks)    

                for label, landmarks in saved_landmarks.items():
                    match = True
                    for saved, detected in zip(landmarks, detected_landmarks):
                    
                        if weighted_distance(saved, detected) > threshold: 
                      
                            match = False
                            break
                    
                    if match:
                        gesture_matched = True 
                        recent_detections.append(label) 
                        
                        
                        most_common = max(set(recent_detections), key=recent_detections.count) 
                        if recent_detections.count(most_common) >= history_frames * 0.6:  
                            detected_gesture = most_common
                        else:
                            detected_gesture = None
                           
                        if detected_gesture and detected_gesture == label:

                            if gesture_start_time is None:
                                gesture_start_time = time.time()
                            elif time.time() - gesture_start_time >= hold_time:
                                confirmed_gesture = label  
                                print(f"Gesture confirmed: {label}")
                                gesture_start_time = None  
                                if label == "finish":
                                    print(f"Word Confirmed: {''.join(word_spelled)}")
                                    word_spelled = [] 
                                elif label == "backspace":
                                    if word_spelled:
                                        word_spelled.pop()  
                                        print(f"Word after backspace: {''.join(word_spelled)}")
                                else:
                                    word_spelled.append(label)  
                                    print(f"Current word: {''.join(word_spelled)}")
                        else:
                            detected_gesture = label
                            gesture_start_time = time.time()
                        break
                else:
                    detected_gesture = None
                    gesture_start_time = None


        if gesture_matched:
            text = f"Detected: {confirmed_gesture if confirmed_gesture else detected_gesture}"
            color = (0, 255, 0) if confirmed_gesture else (0, 255, 255) 
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            confirmed_gesture = None 

        cv2.imshow("Sawyer Head Camera - Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 46:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    compare_gesture_live(hold_time=1.20)

