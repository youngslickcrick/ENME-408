#! /usr/bin/env python3
# Chris Iheanacho @ UMBC
# Description: Compares live hand gestures using Sawyer's head camera against saved gestures
# and combines them into a word.
# ysc_mediapipe_asl_ros2.py


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

"""
rospy: Python client library for ROS that enables programmers to communicate with ROS topics, services, and parameters. In other words it uses the ROS python API for communication with Sawyer.
cv2(openCV): Imports the openCV library which provides functions for and classes for image Image processing and computer vision. It’s essential in opening up sawyer’s cameras.
os (operating systems): This allows for file manipulation and management, this is needed when saving images and javascript object notation (json) files to a specific directory.
json(Javascript object notation): Used to store and transfer data, in this code it’s used to save the coordinates of the hand landmarks in JSON format.
mediapipe: This is used for the hand tracking.
intera_interface: The python API for communicating with Intera-enabled robots
cv_bridge: The bridge between ROS image messages and openCV image representation
time: Is used for handling time based operations it’s used when measuring how long a gesture has been held.
numpy: Is used to call mathematical equations in multiple functions
deque: Is used for storing past frames, helps improve the accuracy of the gestures being by comparing the last few frames and making sure most of them detect the last gesture.
"""


#Global Variables (variables that can be accessed in and out of a function)
bridge = CvBridge()
latest_frame = None
currentlb = 0

#FROM MEDIAPIPE TEMPLATE: Mediapipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, 
                       max_num_hands=1,
                       min_detection_confidence=0.5, 
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


#Opens and closes the sawyer gripper
def grip(close=False):
    gripper = intera_interface.Gripper()
    
    if close:
        gripper.close()       
    else:
        gripper.open()
        
        
#FROM RETHINKROBOTICS TEMPLATE:Processes images from Sawyer's head camera
def camera_callback(img_data, camera_name):
    global latest_frame
    try:
        frame = bridge.imgmsg_to_cv2(img_data, "bgr8")

        latest_frame = cv2.flip(frame, 1)

    except CvBridgeError as e:
        rospy.logerr(f"CV Bridge Error: {e}")

#Normalizes landmarks to the wrist landmark
def normalize_landmarks(landmarks): 

    wrist = landmarks[0]  
    normalized_landmarks = [] 
    
    for lm in landmarks:
        norm_x = lm['x'] - wrist['x']
        norm_y = lm['y'] - wrist['y']
        norm_z = lm['z'] - wrist['z']
        normalized_landmarks.append({'x': norm_x, 'y': norm_y, 'z': norm_z})

    return normalized_landmarks

#Attributes a weighted system to the coordinates
def weighted_distance(saved, detected):
    weight_x = 1.0 
    weight_y = 1.0  
    weight_z = 0.5 

  
    return np.sqrt(
        (weight_x * (saved['x'] - detected['x']))**2 +
        (weight_y * (saved['y'] - detected['y']))**2 +
        (weight_z * (saved['z'] - detected['z']))**2)


#Displays translated gestures on sawyer head screen
def create_display_image1(detected_gesture, word_spelled):

    width, height = 1024, 600  
    image = np.zeros((height, width, 3), dtype=np.uint8) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 5
    color = (255, 255, 255)  

    
    text_detected = f"Detected: {detected_gesture}" if detected_gesture else "Detected: None"
    cv2.putText(image, text_detected, (50, 200), font, font_scale, color, thickness)

    
    text_word = f"Word: {''.join(word_spelled)}"
    cv2.putText(image, text_word, (50, 400), font, font_scale, color, thickness)
    
    return image
    
    
#Displays joint movement on sawyer head screen
def create_display_image2(detected_gesture, word_spelled):
    
    global joint
    width, height = 1024, 600  
    image = np.zeros((height, width, 3), dtype=np.uint8) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 4
    color = (255, 255, 255)  

    
    text_detected = f"Detected: {detected_gesture}" if detected_gesture else "Detected: None"
    cv2.putText(image, text_detected, (50, 200), font, font_scale, color, thickness)

    
    text_word = f"Controlling joint: {joint}"
    cv2.putText(image, text_word, (50, 400), font, font_scale, color, thickness)
    
    return image
    
def head_pan(newlb):
    global currentlb
    
    hp = intera_interface.Head()
    lb = intera_interface.Limb()

    currenthp = hp.pan()
    
    diff = currentlb - newlb
    ratio = 0.96098
    #pan_mode()
    hp.set_pan(currenthp+diff*ratio, speed=0.5)

    


#Compares the saved coordinates of landmarks to the new ones detected by the camera
def compare_gesture_live(threshold=0.06, hold_time=1.0, history_frames=5):

    global joint
    SAVED_GESTURES_PATH = "/home/ysc/ros_ws/src/intera_sdk/intera_examples/scripts/captured_gestures_ros"
    saved_landmarks = {}
    gesture_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "1","2","3","4","5","6","7","8","9" ,"finish", "backspace"]  

    for label in gesture_labels:
        json_path = os.path.join(SAVED_GESTURES_PATH, f"{label}_asl_landmarks_ros.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                saved_landmarks[label] = normalize_landmarks(json.load(f))
    #Intializes rospy nodes for running
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
 
    joint = None
    joint_control_mode = False
    head_display = intera_interface.HeadDisplay()
    gesture_start_time = None
    detected_gesture = None
    confirmed_gesture = None
    recent_detections = deque(maxlen=history_frames)  
    word_spelled = []
    limb = intera_interface.Limb()
    currentlb = limb.joint_angle('right_j0')
    
    
    #Joint angles for different positions
    C1 = {'right_j0': 0.1262578125, 'right_j1': 0.422787109375, 'right_j2': 0.2639326171875, 'right_j3': -0.4753544921875, 'right_j4': 2.9767138671875, 'right_j5': -1.6180185546875, 'right_j6': -0.989341796875}
    
    O = {'right_j0': 0.0, 'right_j1': 0.0, 'right_j2': 0.0, 'right_j3': -0.0, 'right_j4': 0.0, 'right_j5': 0.0, 'right_j6': -0.0}

    C3 = {'right_j0': 0.519818359375, 'right_j1': 0.24602734375, 'right_j2': 0.2218564453125, 'right_j3': -0.1082734375, 'right_j4': 2.9670185546875, 'right_j5': -1.3883837890625, 'right_j6': -0.93182421875}



    #Keeps the function looping as long ROS is active
    while not rospy.is_shutdown():
        if latest_frame is None:
            continue

        frame = latest_frame.copy()

        #FROM MEDIAPIPE TEMPLATE: Used to convert frame to RGB which mediapipe runs in
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        gesture_matched = False  

        #Creates a green box on the camera feed screen
        height, width = frame.shape[:2]
        box_size = int(height * 0.2)
        top_left = (width // 2 - box_size // 2, int(height * 0.25))  
        bottom_right = (top_left[0] + box_size, top_left[1] + box_size)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        #FROM MEDIAPIPE TEMPLATE: Draws the landmarks on the hand
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
              
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

                #Saves detected landmarks as a dictonary and normalizes them
                detected_landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
                detected_landmarks = normalize_landmarks(detected_landmarks)   
                 
                 
                #Loops through letters till it finds a match that meets threshold
                for label, landmarks in saved_landmarks.items():
                    match = True
                    for saved, detected in zip(landmarks, detected_landmarks):
                        if weighted_distance(saved, detected) > threshold: 
                            match = False
                            break
                    
                    #Adds latest detected letter to recent_dectections
                    if match:
                        gesture_matched = True  
                        recent_detections.append(label)  
                        
                        #Determines which dectections shows up the most frequently in recent_detections
                        most_common = max(set(recent_detections), key=recent_detections.count) 
                        if recent_detections.count(most_common) >= history_frames * 0.6:  
                            detected_gesture = most_common
                        else:
                            detected_gesture = None
                           
                        #Starts hold time for hand gesture   
                        if detected_gesture and detected_gesture == label:
                            if gesture_start_time is None:
                                gesture_start_time = time.time()
                            elif time.time() - gesture_start_time >= hold_time:
                                confirmed_gesture = label  
                                print(f"Gesture confirmed: {label}")
                                gesture_start_time = None  
                                
                                #Confirms spelled word                        
                                if label == "finish":
                                    print(f"Word Confirmed: {''.join(word_spelled)}")
                                    hold_time = 1.2
                                    joint_control_mode = False
                                    joint = None
                                    confirmed_gesture = None
                                    gesture_start_time = None
                                    
                                    sword = ''.join(word_spelled)

                                    """
                                    if sword == 'B':
                                        limb.move_to_joint_positions(C1)
                                    elif sword == 'A':
                                        limb.move_to_joint_positions(C3)
                                        #Zero
                                    elif sword == 'Q':
                                        limb.move_to_joint_positions(O)
                                        
                                        #Grip
                                    elif sword == 'Grip':
                                        grip(close=True)
                                        #Ungrip
                                    elif sword == 'Ungrip':
                                        grip(close=False) 
                                    """   

                                   
                                    #After word is confirmed, new the list is emptied  
                                    word_spelled = []


                                #Deletes last letter  
                                elif label == "backspace":
                                    joint = None
                                    if word_spelled:
                                        word_spelled.pop()  
                                        print(f"Word after backspace: {''.join(word_spelled)}")
                                

                                #Combines latest letter to last letter        
                                else:
                                    word_spelled.append(label)  
                                    print(f"Current word: {''.join(word_spelled)}")
                                   
                                    charac = confirmed_gesture
                                    
                                    if charac == 'A':
                                        limb.move_to_joint_positions(C3)

                                    elif charac == 'B':
                                        limb.move_to_joint_positions(O)
                                    #Zero
                                    elif charac == 'C':
                                        limb.move_to_joint_positions(C1)
                                      
                                        
                                    #Grip
                                    if charac == 'G':
                                        grip(close=True)
                                    #Ungrip
                                    elif charac == 'V':
                                        grip(close=False)
                                    
 
                                    elif charac == 'L':
                                        print("Joint control mode: Select joint number")

                                        joint_control_mode = True
                                        hold_time = 0.0
                                        gesture_start_time = time.time()
                                                   
                                    elif joint_control_mode and joint is None and charac in ['O', '1', '2', '3', '4', '5', '6']:
                                        number = '0' if charac == 'O' else charac
                                        joint = f'right_j{number}'
                                        print(f"Controlling joint: {joint}")
                                        print(" Use 'I' to increase, 'D' to decrease, 'finish' to exit.")
                                    elif joint_control_mode and confirmed_gesture in ["I", "D", "finish"]:
                                        if label == "finish":
                                            print("Exiting joint control mode.")
                                            hold_time = 1.0
                                            joint_control_mode = False
                                            joint = None
                                            confirmed_gesture = None
                                            gesture_start_time = None
                                            

                                        else:
                                            delta = 0.01 if confirmed_gesture == "I" else -0.01
                                            current_position = limb.joint_angle(joint)
                                            new_position = current_position + delta
                                            limb.move_to_joint_positions({joint: new_position}, test=True)
                                            #newlb = limb.joint_angle('right_j0')
                                            #head_pan(newlb)
                                            print(f"{confirmed_gesture} {joint} to {new_position}")
                                            time.sleep(0.003)
                                      



                                            
                        #Resets the loop
                        else:                         
                            detected_gesture = label
                            gesture_start_time = time.time()
                        break
                        
                else:
                    detected_gesture = None
                    gesture_start_time = None
                    
                    
                    
        #Prints confirmed and detected letter on camera feed 
        if gesture_matched:
            text = f"Detected: {confirmed_gesture if confirmed_gesture == label and detected_gesture == label else detected_gesture}"
            if confirmed_gesture == label and detected_gesture == label:
                color = (0, 255, 0)
            else:
                color = (0, 255, 255)

            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            confirmed_gesture = None  


        #Opens the camera feed
        cv2.imshow("Sawyer Head Camera - Gesture Recognition", frame)
        
        #Displays image to sawyer head screen
        
        if joint_control_mode == True:
            display_img = create_display_image2(detected_gesture, word_spelled)
            temp_image_path = "/tmp/sawyer_display.png"
            cv2.imwrite(temp_image_path, display_img)
            head_display.display_image(temp_image_path, display_in_loop=False, display_rate=10.0) 
             
        else:
            display_img = create_display_image1(detected_gesture, word_spelled)
            temp_image_path = "/tmp/sawyer_display.png"
            cv2.imwrite(temp_image_path, display_img)  
            head_display.display_image(temp_image_path, display_in_loop=False, display_rate=10.0) 
              
        
        #Closes script if '.' is pressed
        if cv2.waitKey(1) & 0xFF == 46:
            break
            
    cv2.destroyAllWindows()

#Used as guard for against other scripts allowing it to run properly if called
if __name__ == "__main__":
    compare_gesture_live() 
