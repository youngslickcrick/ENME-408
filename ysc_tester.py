#! /usr/bin/env python3
# Chris Iheanacho @ UMBC
# Description: Compares live hand gestures using Sawyer's head camera against saved gestures
# and combines them into a word.
# ysc_mediapipe_asl_ros3.py


# Section 1: Libary
# ------------------------------------------------------------------------------------------

# rospy: Python client library for ROS that enables programmers to communicate with ROS topics, services, and parameters. In other words it uses the ROS python API for communication with Sawyer.
# cv2(openCV): Imports the openCV library which provides functions for and classes for image processing and computer vision.
# os(operating systems): This allows for file manipulation and management, this is needed when saving images and javascript object notation (json) files to a specific directory.
# json(Javascript object notation): Used to store and transfer data.
# time: Is used for handling time based operations it’s used when measuring how long a gesture has been held.
# numpy: Is used to call mathematical equations in multiple functions
# mediapipe: This is used for the hand tracking.
# intera_interface: The python API for communicating with Intera-enabled robots
# deque: Is used for storing past frames, helps improve the accuracy of the gestures being by comparing the last few frames and making sure most of them detect the last gesture.
# Cvbridge: The bridge between ROS image messages and openCV image representation
# CvBridgeError: Used for handling errors in openCV
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
import threading



# Section 2: Global Variables and Mediapipe intialization
# ---------------------------------------------------------------------------------------------

# Variables that can be accessed in and out of a function
bridge = CvBridge()
latest_frame = None
currentlb = 0
increment = 0
incr = None
change_increment = False
pos_mode = True
number = 0
save_mode = False
execution_mode = False
charac = None
text_word = None
text_word1 = None
text_word2 = None
save_word = None
traj_mode = False
coolguy = False
joint = False
joint_word = None
joint_control_mode = False
traj_mode = False
gametimer = False
gesture_pool = []
timer_running = False
stop_timer = False
elapsed = 0
sword = None
timer_off = False
#FROM MEDIAPIPE TEMPLATE: Mediapipe settings, decides the accuracy and mode of the landmarks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, 
                       max_num_hands=1,
                       min_detection_confidence=0.5, 
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils




# Section 3: Basic Functions
# ---------------------------------------------------------------------------------------------
# Functions that will be accessed in and outside of the main function compare_gestures_live()

#Opens and closes the sawyer gripper
def grip(close=False):
    #Using the gripper class from intera_interface
    gripper = intera_interface.Gripper()
    
    if close:
        gripper.close()       
    else:
        gripper.open()
        
        
#FROM RETHINKROBOTICS TEMPLATE: Processes images from Sawyer's head camera
def camera_callback(img_data, camera_name):
    #Using 'global' to call this variable
    global latest_frame
  
    #Try block allows one to test code for errors
    try:
        #Converting ros images messages to cv2 for processing
        frame = bridge.imgmsg_to_cv2(img_data, "bgr8")
        #Mediapipe requries a flipped frame
        latest_frame = cv2.flip(frame, 1)
      
    #Except handles the error
    except CvBridgeError as e:
        rospy.logerr(f"CV Bridge Error: {e}")


#Normalizes landmarks to the wrist landmark
def normalize_landmarks(landmarks): 
    #The first landmark located at the wrist
    wrist = landmarks[0]  
    #Creates an empty list to intialize 
    normalized_landmarks = [] 
    
    #Calls each landmark for this equation
    for lm in landmarks:
        norm_x = lm['x'] - wrist['x']
        norm_y = lm['y'] - wrist['y']
        norm_z = lm['z'] - wrist['z']
        normalized_landmarks.append({'x': norm_x, 'y': norm_y, 'z': norm_z})
    #Returns the value of the calculated function
    return normalized_landmarks


#Attributes a weighted system to the coordinates
def weighted_distance(saved, detected):

    weight_x = 1.0 
    weight_y = 1.0  
    #Z is the distance away from the camera
    weight_z = 0.5 

    #Using the weights in tangent with Euclidean distance formula to measure the distance between saved and detected landmarks
    return np.sqrt(
        (weight_x * (saved['x'] - detected['x']))**2 +
        (weight_y * (saved['y'] - detected['y']))**2 +
        (weight_z * (saved['z'] - detected['z']))**2)


# Create images for posistion mode
def create_display_image1(detected_gesture, word_spelled):
    global elapsed, gametimer, timer_off
    #1024x600 pixels
    width, height = 1024, 600  
    
    #np.full(dimensions, color, dtype), the alternative would be np.zeroes(dimensions, dtype)
    #Used to create blank images where text can be added
    #(0,0,0) = black, (255,255,255) = white
    #dtype is the range in which the pixels will be defined, uint8 defines the range as 0-255 which is standard for image
    image = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8) 


    font = cv2.FONT_HERSHEY_SIMPLEX
    font2 = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.7
    thickness = 3
    thickness2 = 2
    color = (0, 0, 0)  

    timer = f"Time: {elapsed} secs   Score:{max(0, int(10000/((elapsed +1))))} pts" 
    
    if timer_off:
        cv2.putText(image, "", (50, 400), font, 1.4, color, thickness)
    else:
        cv2.putText(image, timer, (50, 400), font, 1.4, color, thickness)
    
    #shortened if else statement
    text_detected = f"Detected: {detected_gesture}" if detected_gesture else "Detected: None"
    cv2.putText(image, text_detected, (50, 200), font, font_scale, color, thickness)

    
    text_word = f"Word: {''.join(word_spelled)}"
    cv2.putText(image, text_word, (50, 300), font, font_scale, color, thickness)
    
    key1 = "'L' for Joint Mode"
    key2 = "'B'+'finish' to start and stop timer"
    key4 = "'S' for Save Mode"
    key5 = "'E' for Execution Mode"
    key6 = "'Y' for Trajectory Mode"
    key7 = "'G' to close gripper"
    key8 = "'U' to open gripper"
    key9 = "'finish' to confirm word"
    key10 = "'backspace' to delete character"

    cv2.putText(image, key1, (50, 50), font, 1, (255,0,0), 2)
    cv2.putText(image, key2, (50, 120), font, 1, (0,0,0), 2)
    cv2.putText(image, key4, (50, 85), font, 1, (0,180,0), 2)
    cv2.putText(image, key5, (600, 50), font, 1, (0,0,255), 2)
    cv2.putText(image, key6, (600, 85), font, 1, (0,175,255), 2)
    cv2.putText(image, key7, (50, 465), font, 1, (0,0,0), 2)
    cv2.putText(image, key8, (50, 500), font, 1, (0,0,0), 2)
    cv2.putText(image, key9, (50, 535), font, 1, (0,0,0), 2)
    cv2.putText(image, key10, (50, 570), font, 1, (0,0,0), 2)
    
    #Returns the image so that it can be displayed
    return image
    
    
# Create images for joint control mode
def create_display_image2(detected_gesture):
    global incr, joint, joint_word
    global elapsed, gametimer, timer_off
    global charac, save_mode, execution_mode
    
    width, height = 1024, 600  
    image = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.7
    thickness = 3
    color = (0, 0, 0)  

    text_detected = f"Detected: {detected_gesture}" if detected_gesture else "Detected: None"
    cv2.putText(image, text_detected, (50, 200), font, font_scale, color, thickness)


    text_word = f"Controlling joint: {joint}"
    cv2.putText(image, text_word, (50, 300), font, font_scale, (255, 0, 0), thickness)
    
    timer = f"Time: {elapsed} secs   Score:{max(0, int(10000/((elapsed +1))))} pts"

    if timer_off:
        cv2.putText(image, "", (50, 400), font, 1.4, color, thickness)
    else:
        cv2.putText(image, timer, (50, 400), font, 1.4, color, thickness)
    
    if increment == 0.5:
        incr = "Large"
        num = "0.5"
    elif increment == 1.0:
        incr = "Huge"
        num = "1.0"
    elif increment == 0.01:
        incr = "Small"
        num = "0.01"
    else:
        incr = "Default"
        num = "0.25"
        
    if joint == None:
        joint_word = "Select joint number: 0-6"
    else:
        joint_word = "'P' to change increment"
      
    key7 = "'G' to close gripper"
    key8 = "'U' to open gripper"
    key9 = "'finish' to exit to home screen"
    key10 = "'backspace' to change joints"
    
  
    
    if save_mode == True:
        execution_mode = False
        cv2.putText(image, f"Save Mode On", (600, 110), font, 1, (0,230,0), 2)
        if charac in ['1','2','3','4','5','6','7','8','9']:
            number = charac
            cv2.putText(image, f"File {number} Selected", (600, 145), font, 1, (0,230,0), 2)
        else:
            key2 = "Select File"
            cv2.putText(image, key2, (600, 145), font, 1, (0,230,0), 2)

            
    elif execution_mode == True:
        save_mode = False
        cv2.putText(image, f"Execution Mode On", (600, 110), font, 1, (0,0,255), 2)
        if charac in ['1','2','3','4','5','6','7','8','9']:
            number = charac
            cv2.putText(image, f"File {number} Selected" , (600, 145), font, 1, (0,0,255), 2)
        else:
            key2 = "Select File"
            cv2.putText(image, key2, (600, 145), font, 1, (0,0,255), 2)


    
    cv2.putText(image, key7, (600, 500), font, 1, (0,0,0), 2)
    cv2.putText(image, key8, (600, 535), font, 1, (0,0,0), 2)
    cv2.putText(image, key9, (50, 500), font, 1, (0,0,0), 2)
    cv2.putText(image, key10, (50, 535), font, 1, (0,0,0), 2)  
      
    text_word = f"Increment: {incr} ({num} rad)"
    cv2.putText(image, text_word, (50, 460), font, 1, (255, 0, 0), 2)        
    cv2.putText(image, joint_word, (50, 110), font, 1, (255, 0, 0), 2)
        
    return image
    
# Create images for changing the increment
def create_display_image3(detected_gesture):
    global incr, increment
    global elapsed, gametimer, timer_off
        
    width, height = 1024, 600  
    image = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.7
    thickness = 3
    color = (0, 0, 0)  

    text_detected = f"Detected: {detected_gesture}" if detected_gesture else "Detected: None"
    cv2.putText(image, text_detected, (50, 200), font, font_scale, color, thickness)


    cv2.putText(image, "Change Increment", (50, 300), font, font_scale, (255, 0, 0), thickness)
    
    timer = f"Time: {elapsed} secs   Score:{max(0, int(10000/((elapsed +1))))} pts"
    if timer_off:
        cv2.putText(image, "", (50, 400), font, 1.4, color, thickness)
    else:
        cv2.putText(image, timer, (50, 400), font, 1.4, color, thickness)
        
    cv2.putText(image, "Sign 'R' for huge, 'F' for large or 'W' for small", (50, 470), font, 1, (255, 0, 0), 2)
    cv2.putText(image, "'R' = 1.0 rad, 'F' = 0.5 rad, or 'W' = 0.01 rad", (50, 505), font, 1, (255, 0, 0), 2)
    
    key10 = "'backspace' to exit change increment screen"
    cv2.putText(image, key10, (50, 540), font, 1, (0,0,0), 2) 
    
    key9 = "'finish' to exit to home screen"
    cv2.putText(image, key9, (50, 575), font, 1, (0,0,0), 2)
    
    return image    

# Create images for save mode or saved angles mode
def create_display_image4(detected_gesture):
    global save_mode
    global execution_mode
    global charac
    global text_word
    global text_word1
    global text_word2
    global save_word
    global timer_off, elapsed
  
    width, height = 1024, 600  
    image = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.7
    thickness = 3
    color = (0, 0, 0)  

    
    text_detected = f"Detected: {detected_gesture}" if detected_gesture else "Detected: None"
    cv2.putText(image, text_detected, (50, 200), font, font_scale, color, thickness)

    timer = f"Time: {elapsed} secs   Score:{max(0, int(10000/((elapsed +1))))} pts"
    if timer_off:
        cv2.putText(image, "", (50, 400), font, 1.4, color, thickness)
    else:
        cv2.putText(image, timer, (50, 400), font, 1.4, color, thickness)
    
    if save_mode == True:
        save_word = f"Save mode, select save file: 1-9"
        colour = (0, 180, 0)
        execution_mode = False
        key = "'E' for Execution Mode"
        col = (0,0,255)
        if charac in ['1','2','3','4','5','6','7','8','9']:
            number = charac
            text_word1 = "Current joint angles saved in" 
            text_word2 = f"saved_joint_angles_{number}"
        else:
            text_word1 = "Waiting for a number gesture..."
            text_word2 = ""
       
        
    elif execution_mode == True:
        save_word = f"Saved Angles Execution Mode, select saved file: 1-9"
        colour = (0, 0, 255)
        save_mode = False
        key = "'S' for Save Mode"
        col = (0,230,0)
        if charac in ['1','2','3','4','5','6','7','8','9']:
            number = charac
            text_word1 = "Moving to" 
            text_word2 = f"saved_joint_angles_{number}"
        else:
            text_word1 = "Waiting for a number gesture..."
            text_word2 = ""
            
    cv2.putText(image, save_word, (50, 110), font, 1, colour, 2)
    cv2.putText(image, text_word1, (50, 270), font, font_scale, color, thickness)
    cv2.putText(image, text_word2, (50, 330), font, font_scale, color, thickness)    
    

    key9 = "'Y' for Trajectory Mode"

    cv2.putText(image, key, (50, 500), font, 1, col, 2)
    cv2.putText(image, key9, (50, 535), font, 1, (0,175,255), 2)
            
    key9 = "'finish' to exit to home screen"
    cv2.putText(image, key9, (50, 570), font, 1, (0,0,0), 2)
    
    return image    


# Create images for trajectory mode
def create_display_image5(detected_gesture, word_spelled):
    global timer_off, elapsed
    width, height = 1024, 600  
    image = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.7
    thickness = 3
    color = (0, 0, 0)  

    text_detected = f"Detected: {detected_gesture}" if detected_gesture else "Detected: None"
    cv2.putText(image, text_detected, (50, 200), font, font_scale, color, thickness)

    sword = deque(maxlen=5)
    sword = ''.join(word_spelled)

    if sword == 5:
        nl = ''.join(word_spelled)
    else:
        nl = ''
      
    text_word = f"Path: {sword}"
    cv2.putText(image, text_word, (50, 300), font, font_scale, color, thickness)
    cv2.putText(image, nl, (50, 450), font, font_scale, color, thickness)

    traj_word = "Trajectory Mode: Select saved files and sign 'M' to execute"
    cv2.putText(image, traj_word, (50, 100), font, 1, (0, 175, 255), 2)
   
   
    key4 = "'S' for Save Mode"
    key5 = "'E' for Execution Mode"
    cv2.putText(image, key4, (600, 465), font, 1, (0,180,0), 2)
    cv2.putText(image, key5, (600, 500), font, 1, (0,0,255), 2)
    
    key1 = "'G' to add close gripper to path"
    key2 = "'U' to add open gripper to path"
    cv2.putText(image, key1, (50, 465), font, 1, (0,0,0), 2)
    cv2.putText(image, key2, (50, 500), font, 1, (0,0,0), 2)

    key9 = "'finish' to exit to home screen"
    cv2.putText(image, key9, (50, 535), font, 1, (0,0,0), 2)

    timer = f"Time: {elapsed} secs   Score:{max(0, int(10000/((elapsed +1))))} pts"
    if timer_off:
        cv2.putText(image, "", (50, 400), font, 1.4, color, thickness)
    else:
        cv2.putText(image, timer, (50, 400), font, 1.4, color, thickness)        
    return image

def create_display_image6(detected_gesture, word_spelled):

    width, height = 1024, 600  
    image = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.7
    thickness = 3
    color = (0, 0, 0)  

    text_detected = f"Detected: {detected_gesture}" if detected_gesture else "Detected: None"
    cv2.putText(image, text_detected, (50, 200), font, font_scale, color, thickness)

    sword = deque(maxlen=5)
    sword = ''.join(word_spelled)

    if sword == 5:
        nl = ''.join(word_spelled)
    else:
        nl = ''
      
    text_word = f"Path: {sword}"
    cv2.putText(image, text_word, (50, 300), font, font_scale, color, thickness)
    cv2.putText(image, nl, (50, 450), font, font_scale, color, thickness)

    traj_word = "Name for score"
    cv2.putText(image, traj_word, (50, 100), font, 1, (0, 175, 255), 2)
   
   
    key4 = "'S' for Save Mode"
    key5 = "'E' for Execution Mode"
    cv2.putText(image, key4, (600, 465), font, 1, (0,180,0), 2)
    cv2.putText(image, key5, (600, 500), font, 1, (0,0,255), 2)
    
    key1 = "'G' to add close gripper to path"
    key2 = "'U' to add open gripper to path"
    cv2.putText(image, key1, (50, 465), font, 1, (0,0,0), 2)
    cv2.putText(image, key2, (50, 500), font, 1, (0,0,0), 2)

    key9 = "'finish' to exit to home screen"
    cv2.putText(image, key9, (50, 535), font, 1, (0,0,0), 2)

    timer = f"Timer: {elapsed} secs" if gametimer == True else "Timer: 0 secs"
    cv2.putText(image, timer, (50, 400), font, font_scale, color, thickness)        
    return image

def game_timer():
    global timer_running, stop_timer, elapsed
    start_time = time.perf_counter()
    timer_running = True
    while not stop_timer:
        elapsed = int(time.perf_counter() - start_time+1)
        #print(f"Timer: {elapsed} second(s)")
        time.sleep(1)
    timer_running = False
    print("Timer stopped.")

def start_game_timer():
    global stop_timer
    stop_timer = False
    if not timer_running:
        threading.Thread(target=game_timer, daemon=True).start()

def stop_game_timer():
    global stop_timer
    stop_timer = True

#EXPERIMENTAL CODE
# Normalizes the head pan to joint0 axis
"""
def head_pan(newlb):
    global currentlb
    
    hp = intera_interface.Head()
    lb = intera_interface.Limb()

    currenthp = hp.pan()
    
    diff = currentlb - newlb
    ratio = 0.96098
    #pan_mode()
    hp.set_pan(currenthp+diff*ratio, speed=0.5)
"""
    

# Section 4: Main Function
# ---------------------------------------------------------------------------------------------
# Compares the saved coordinates of landmarks to the new ones detected by the camera

# MODES
#---------------------------
# DEFAULT MODE: Move to joint angles that are saved directly in the code (Black)
# JOINT CONTROL MODE: Individually control each joint by increasing or decreasing its joint angle by a set increment, (Increment can be changed with change increment mode) (Blue)
# SAVE MODE: Save the current joint angles of the robot in 1 of 9 files (Green)
# SAVED ANGLES EXECUTION MODE: Execute the saved joint angles individually (Red)
# TRAJECTORY MODE: Execute the saved joint angles in succession creating a path for the robot (Orange)

def compare_gesture_live(threshold=0.06, hold_time=1.0, history_frames=5):
    global increment
    global joint
    global incr
    global change_increment
    global pos_mode
    global save_mode
    global execution_mode
    global charac 
    global coolguy
    global joint_control_mode
    global gesture_pool
    global gametimer
    global elapsed
    global timer_off
    global sword
    # Initializes a ROS node for running
    rospy.init_node("sawyer_gesture_recognition", anonymous=True) 
    
    
    # Variable initialization
    delta = 0
    gripper = True
    joint = None 
    gesture_start_time = None
    detected_gesture = None
    confirmed_gesture = None
    head_display = intera_interface.HeadDisplay()
    limb = intera_interface.Limb()
    limb.__init__(limb ='right', synchronous_pub=False)
    currentlb = limb.joint_angle('right_j0')
    # Sets recent_detections to have a length 5 max
    recent_detections = deque(maxlen=history_frames)  
    word_spelled = []
    traj_mode = False
    naming = False
    #joint_control_mode = False
    
    
    
    # Define the path where the saved gesture landmark files are stored
    SAVED_GESTURES_PATH = "/home/ysc/ros_ws/src/intera_sdk/intera_examples/scripts/captured_gestures_ros"
    # Initialize a dictionary to store the loaded and normalized landmarks
    saved_landmarks = {}
    # Defining which gesture labels will be utilized
    letter_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]  
    pos_labels = ["A", "B", "C", "E", "G", "L", "S", "U", "Y", "Q" ,"finish", "backspace", "midfi"] 
    number_labels = ["1","2","3","4","5","6","7","8","9"]
    control_labels = ["finish", "backspace", "midfi", "G", "U"]
    cl = ["backspace","finish"]
    yes = ["Y", "E","S","L"]
    joint_labels = ["O","1","2","3","4","5","6","P"]
    joint_labels_control = ["P", "I","D"]
    save_labels = ["S"]
    execution_labels = ["E"]
    increment_labels = ["W","F","R"]
    traj_labels = ["M"]
  
    gesture_pool = pos_labels
    
    # For every label in gesture labels
    for label in gesture_pool:
        json_path = os.path.join(SAVED_GESTURES_PATH, f"{label}_asl_landmarks_ros.json")
        # Checks if the file exists
        if os.path.exists(json_path):
            # Opens and the file for reading
            with open(json_path, 'r') as f:
                # Normalize the loaded landmark data and store it in the saved_landmarks dictionary using the label as the key
                saved_landmarks[label] = normalize_landmarks(json.load(f))
                
    # Defines cameras as sawyers Camera class
    cameras = intera_interface.Cameras()
    camera_name = "head_camera"

    # Verifies camera exists otherwise will throw an error
    if not cameras.verify_camera_exists(camera_name):
        rospy.logerr(f"Error: Camera '{camera_name}' not detected. Exiting.")
        return

    rospy.loginfo(f"Starting stream from {camera_name}...")
    # Opens head camera
    cameras.start_streaming(camera_name)

    # Camera settings
    cameras.set_gain(camera_name, -1)
    cameras.set_exposure(camera_name, -1)

    # #FROM RETHINKROBOTICS TEMPLATE: Callback method used to show the camera image
    cameras.set_callback(camera_name, camera_callback, rectify_image=True, callback_args=(camera_name,))
 
        
    # Joint angles for different in default mode positions
    C1 = {'right_j0': 0.1262578125, 'right_j1': 0.322787109375, 'right_j2': 0.2639326171875, 'right_j3': -0.4753544921875, 'right_j4': 2.9767138671875, 'right_j5': -1.6180185546875, 'right_j6': -0.989341796875}
    
    O = {'right_j0': 0.0, 'right_j1': 0.0, 'right_j2': 0.0, 'right_j3': -0.0, 'right_j4': 0.0, 'right_j5': 0.0, 'right_j6': -0.0}

    C3 = {'right_j0': 0.519818359375, 'right_j1': 0.24602734375, 'right_j2': 0.2218564453125, 'right_j3': -0.1082734375, 'right_j4': 2.9670185546875, 'right_j5': -1.3883837890625, 'right_j6': -0.93182421875}

    mf = {"right_j0": 0.2555947265625, "right_j1": -1.6592841796875, "right_j2": -3.0424638671875, "right_j3": -0.052869140625, "right_j4": 2.9232890625, "right_j5": 0.00548046875, "right_j6": 0.4243916015625}

    # Keeps the function looping as long ROS is active
    while not rospy.is_shutdown():
        if latest_frame is None:
            continue
        
        if joint_control_mode == True and change_increment == False and joint == None and save_mode == False and execution_mode == False:
            gesture_pool = joint_labels + control_labels + yes
            
        elif joint_control_mode == True and joint != 0 and change_increment == False and save_mode == False and execution_mode == False:
            gesture_pool = joint_labels_control + control_labels + yes
            
        elif change_increment == True and joint_control_mode == True:
            gesture_pool = increment_labels + cl
            
        elif save_mode == True or joint_control_mode == True and save_mode == True:
            gesture_pool = control_labels + number_labels + yes
            
        elif execution_mode == True or joint_control_mode == True and execution_mode == True:
            gesture_pool = control_labels + number_labels + yes
            
        elif traj_mode == True:
            gesture_pool = traj_labels + control_labels + number_labels + yes
            
        elif pos_mode == True and naming == False:
            gesture_pool = pos_labels 
            
        elif naming == True:
            gesture_pool = control_labels + letter_labels
        
        saved_landmarks = {}
        for label in gesture_pool:
            json_path = os.path.join(SAVED_GESTURES_PATH, f"{label}_asl_landmarks_ros.json")
        # Checks if the file exists
            if os.path.exists(json_path):
            # Opens and the file for reading
                with open(json_path, 'r') as f:
                # Normalize the loaded landmark data and store it in the saved_landmarks dictionary using the label as the key
                    saved_landmarks[label] = normalize_landmarks(json.load(f))
        
        #Continously updates the frame
        frame = latest_frame.copy()

        # FROM MEDIAPIPE TEMPLATE: Used to convert frame to RGB which mediapipe runs in
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        gesture_matched = False  

        # Creates a green box on the camera feed screen
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

                # Converts detected landmarks to a list of dictionaries (with x, y, z coordinates) and normalizes them
                detected_landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
                detected_landmarks = normalize_landmarks(detected_landmarks)   
                 
                 
                # Loops through letters till it finds a match that meets threshold
                # Sets the match to true for initialization
                for label, landmarks in saved_landmarks.items():
                    match = True
                    # Zips the saved and detected landmarks then compares the weighted distance to the threshold
                    for saved, detected in zip(landmarks, detected_landmarks):
                        if weighted_distance(saved, detected) > threshold: 
                            match = False
                            break
                    
                    # Adds latest detected letter to recent_dectections
                    if match:
                        gesture_matched = True  
                        recent_detections.append(label)  
                        
                        # Determines which dectections shows up the most frequently in recent_detections
                        # Sets most_common equal to the label that shows up the most frequently in recent_dections
                        most_common = max(set(recent_detections), key=recent_detections.count)
                        # If most_common appears 60% which is 3/5 detections, detected_gesture is set to most_common 
                        if recent_detections.count(most_common) >= history_frames * 0.6:  
                            detected_gesture = most_common
                        else:
                            detected_gesture = None
                           
                        #If detected_gesture exists and eqauls a label the hold time starts
                        if detected_gesture and detected_gesture == label:
                            if gesture_start_time is None:
                                gesture_start_time = time.time()
                            # If the hold_time is met, the gesture is confirmed
                            elif time.time() - gesture_start_time >= hold_time:
                                confirmed_gesture = label  
                                print(f"Gesture confirmed: {label}")
                                gesture_start_time = None  
                                
                                #Confirms spelled word and closes various modes                       
                                if label == "finish":
                                    print(f"Word Confirmed: {''.join(word_spelled)}")
                                    hold_time = 1
                                    threshold = 0.07
                                    joint_control_mode = False
                                    joint = None
                                    confirmed_gesture = None
                                    gesture_start_time = None
                                    pos_mode = True
                                    increment = 0
                                    change_increment = False
                                    save_mode = False
                                    execution_mode = False
                                    gripper = True
                                    traj_mode = False
                                    coolguy = False
                                    naming = False
                                    sword = ''.join(word_spelled)
                                
                                    
                                    #Sets the robot to wave
                                    if sword == 'WV':
                                        w = {"right_j0": -0.00189453125, "right_j1": -1.2478955078125, "right_j2": -0.0020673828125, "right_j3": 0.0055654296875, "right_j4": -0.00079296875, "right_j5": 0.0011416015625, "right_j6": 0.001919921875}
                                        v = {"right_j0": -0.0013798828125, "right_j1": -1.995404296875, "right_j2": -0.001201171875, "right_j3": 0.011466796875, "right_j4": -0.005970703125, "right_j5": 0.0003154296875, "right_j6": 0.001919921875}
                                        
                                        image_path = "/home/ysc/Pictures/Hello.png"
                                        head_display.display_image(image_path, display_in_loop=False, display_rate=10.0)
                                        
                                        limb.set_joint_position_speed(0.3)
                                        limb.move_to_joint_positions(w)
                                        rospy.sleep(.5)
                                        limb.move_to_joint_positions(v)
                                        rospy.sleep(.5)
                                        limb.move_to_joint_positions(w)
                                        rospy.sleep(.5)
                                        limb.move_to_joint_positions(v)
                                        

                                    #Starts the on-screen timer
                                    if sword == 'B' and elapsed > 1:
                                        stop_game_timer()
                                        gametimer = False  
                                        pos_mode = False
                                    #Stops the on-screen timer
                                    elif sword == 'B':
                                        start_game_timer()
                                        gametimer = True                                      

                                    #Resets timer to 0
                                    elif sword == "BB":
                                        elapsed = 0         
                                        
                                    #Removes on-screen timer
                                    elif sword == "BBB" and timer_off == True:
                                        timer_off = False                                    
                                    
                                    #Adds on-screen timer
                                    elif sword == "BBB":
                                        timer_off = True

                                    #Turns flag on to give robot access to all letters
                                    elif sword == "Q":
                                        naming = True
                                   
                                    #After word is confirmed, the list is emptied
                                    word_spelled = []

                                #Turns off save mode or executon mode if joint control mode
                                elif label == "backspace" and save_mode == True and joint_control_mode == True or label == "backspace" and execution_mode == True and joint_control_mode == True:
                                    hold_time = 1.0
                                    save_mode = False
                                    execution_mode = False
                                    #increment = 0
                                    
                                    #Deletes last letter in string
                                    if word_spelled:
                                        word_spelled.pop()  
                                        #print(f"Word after backspace: {''.join(word_spelled)}")


                                #Deletes last letter, resets joint and closes change_increment
                                elif label == "backspace":
                                    joint = None
                                    change_increment = False
                                    save_mode = False
                                    execution_mode = False
                                    #increment = 0
                                    if word_spelled:
                                        word_spelled.pop()  
                                        #print(f"Word after backspace: {''.join(word_spelled)}")
                                

                                #Combines latest letter to last letter        
                                else:
                                    word_spelled.append(label)  
                                    #print(f"Current word: {''.join(word_spelled)}")
                                   
                                    charac = confirmed_gesture
                                  
                                    #If 'A' is signed, arm moves to C3
                                    if charac == 'A' and pos_mode == True and naming == False:
                                        limb.set_joint_position_speed(speed = 0.2)
                                        limb.move_to_joint_positions(C3)                                      
                                    #If 'B' is signed arm moves to the zero position
                                    elif charac == 'B' and pos_mode == True and naming == False:
                                        limb.set_joint_position_speed(speed = 0.2)
                                        limb.move_to_joint_positions(O)
                                    #If 'C' is signed, arm moves to C1
                                    elif charac == 'C' and pos_mode == True and naming == False:
                                        limb.set_joint_position_speed(speed = 0.2)
                                        limb.move_to_joint_positions(C1)

                                        
                                    elif charac == 'midfi' and pos_mode:
                                        image_path = "/home/ysc/Pictures/coolguy.jpeg"
                                        head_display.display_image(image_path, display_in_loop=False, display_rate=10.0)
                                        limb.set_joint_position_speed(speed = 0.2)
                                        limb.move_to_joint_positions(mf)                                                                      
                                        coolguy = True
                                        

                                        
                                    #If 'G' is signed the gripper will close
                                    if charac == 'G' and gripper == True:
                                        grip(close=True)
                                    #If 'U' is signed the gripper will open
                                    elif charac == 'U' and gripper == True:
                                        grip(close=False)
                                    
                                    #If 'L' is signed Joint Control Mode will turn on
                                    elif charac == 'L' and naming == False:
                                        print("Joint control mode: Select joint number")
                                        save_mode = False
                                        execution_mode = False
                                        traj_mode = False
                                        pos_mode = False
                                        joint_control_mode = True
                                        gesture_pool = []

                                    #If Joint Control Mode is on and a joint has been selected 'I' or 'D' can be used to increase or decrease the joint angle respectively
                                    #A joint can be slected by signing a number 0-6
                                    elif joint_control_mode and joint is None and charac in ['O', '1', '2', '3', '4', '5', '6']:
                                        number = '0' if charac == 'O' else charac
                                        joint = f'right_j{number}'
                                        print(f"Controlling joint: {joint}")
                                        print(" Use 'I' to increase, 'D' to decrease, 'finish' to exit.")
                                        hold_time = 0.0
                                        
                                    #If Joint Control Mode is on, the increment at which the joint angle can be altered can be edited if 'P' is signed
                                    elif joint_control_mode and confirmed_gesture in ["P", "F", "W","I","D","R","G","U"]:
                                        if confirmed_gesture == "P":
                                            print("CHANGE INCREMENT")
                                            change_increment = True
                                            hold_time = 1.0

                                        #If 'F' is signed while change increment is on the increment will be changed to large
                                        elif confirmed_gesture == "F" and change_increment == True:
                                            increment = 0.5
                                            sleep = 0.05
                                            print("Increment Changed to Large")
                                            change_increment = False
                                        #If 'W' is signed while change increment is on the increment will be changed to small
                                        elif confirmed_gesture == "W" and change_increment == True:
                                            increment = 0.01
                                            sleep = 0.000001
                                            print("Increment Changed to Small")
                                            change_increment = False
                                        #If 'R' is signed while change increment is on the increment will be changed to huge
                                        elif confirmed_gesture == "R" and change_increment == True:
                                            increment = 1
                                            sleep = 0.05
                                            print("Increment Changed to Huge")
                                            change_increment = False

                                        #If change increment is true the incremental value at which the joint angles move will be changed from default
                                        elif increment == 0.5 and confirmed_gesture in ["I", "D"] or increment == 0.01 and confirmed_gesture in ["I", "D"] or increment == 1.0 and confirmed_gesture in ["I", "D"] :
                                            if confirmed_gesture == "I":
                                                delta = increment 
                                            elif confirmed_gesture == "D":
                                                delta = -1*increment

                                            #Try block used for catching errors
                                            try:
                                                if joint != None:
                                                    #PARTIALLY FROM RETHINKROBOTICS TEMPLATE
                                                    #Current position of chosen joint
                                                    current_position = limb.joint_angle(joint)
                                                    #Defining the new position by adding the increment
                                                    new_position = current_position + delta
                                                    
                                                    #Setting the speed at which the joint will move
                                                    if increment == 0.01:
                                                        limb.set_joint_position_speed(speed = 0.3)
                                                    elif increment == 0.5:
                                                         limb.set_joint_position_speed(speed = 0.3)
                                                    elif increment == 1:
                                                        limb.set_joint_position_speed(speed = 0.3)
                                                    #Moves to new joint angle
                                                    limb.move_to_joint_positions({joint: new_position}, test=True)                             
                                                    print(f"{confirmed_gesture} {joint} to {new_position}")     
                                                
                                                elif change_increment == False and joint != None:
                                                    #Error sent to the terminal if constraints are not met
                                                    rospy.logwarn("Change increment is false. Cannot change increment.")
                                                    
                                                elif joint == None:
                                                    #Error sent to the terminal if constraints are not met
                                                    rospy.logwarn("No joint selected. Cannot move.")
                                                    
                                            except Exception as e:
                                                    #Error sent to the terminal if constraints are not met
                                                    rospy.logerr(f"Joint control error: {e}")
                                            time.sleep(sleep) 
                                          
                                        #The incremental value at which the joint angles move will be set to default
                                        elif change_increment == False and confirmed_gesture in ["I","D"]:
                                            if confirmed_gesture == "I":
                                                delta = 0.25 
                                            elif confirmed_gesture == "D":
                                                delta = -0.25
                                                
                                            try:
                                                if joint != None:
                                                    current_position = limb.joint_angle(joint)
                                                    new_position = current_position + delta
                                                    limb.set_joint_position_speed(speed = 0.1)
                                                    limb.move_to_joint_positions({joint: new_position}, test=True)
                                                    print(f"{confirmed_gesture} {joint} to {new_position}")
                                                    
                                                elif change_increment == False and joint != None:
                                                    rospy.logwarn("Change increment is false. Cannot change increment.")
                                                    
                                                elif joint == None:
                                                    rospy.logwarn("No joint selected. Cannot Move.")
                                                    
                                            except Exception as e:
                                                rospy.logerr(f"Joint control error: {e}")
                                            time.sleep(0.01)

                                            #EXPERIMENTAL CODE
                                            #newlb = limb.joint_angle('right_j0')
                                            #head_pan(newlb)
                                            
                                    #If 'S' is signed Save Mode will turn on
                                    elif charac == 'S' and naming == False:
                                        save_mode = True
                                        pos_mode = False
                                        execution_mode = False
                                       # joint_control_mode = False
                                        change_increment == False
                                        hold_time = 1
                                        print('Entering Save Mode, select save file')

                                    #If save mode is on select a number 1-9 to save the joint angles to that designated file
                                    elif save_mode and charac in ['1','2','3','4','5','6','7','8','9']:
                                        number = charac
                                        j_a = limb.joint_angles()
                                        #Opens file based on number signed and writes the joint angles in json format to file
                                        with open(f"saved_joint_angles_{number}","w") as f:
                                            json.dump(j_a, f)
                                            print(f'Joint angles saved to file {number}')

                                    #If 'E' is signed Saved Angles Execution Mode will turn on
                                    elif charac == 'E' and naming == False:
                                        execution_mode = True
                                        save_mode = False
                                        pos_mode = False
                                        #joint_control_mode = False
                                        change_increment == False
                                        hold_time = .8
                                        print('Entering Saved Angles Mode, select saved file')

                                    #If save angles mode is on select a number 1-9 to open the saved file and move the joint angles to
                                    elif execution_mode and charac in ['1','2','3','4','5','6','7','8','9']:
                                        number = charac
                                        #Opens file based on number signed and reads the joint angles
                                        with open(f"saved_joint_angles_{number}", "r") as f:
                                            x = json.load(f)
                                            print(f'Moved to saved angles {number}')
                                        #Executes read file
                                        limb.move_to_joint_positions(x,threshold=0.008726646,test=None)

                                    #If 'Y' is signed Trajectory Mode is turned on
                                    #Select the files a by signing the numbers 1-9, use G and U for closing and opening the gripper
                                    elif charac == 'Y' and naming == False:
                                         traj_mode = True
                                         save_mode = False
                                         execution_mode = False
                                         pos_mode = False
                                         gripper = False
                                         joint_control_mode = False
                                         threshold = 0.07
                                         hold_time = 1.0
                                         word_spelled = []
                                         print('Entering Trajectory Mode: Select Saved Files')
                                         

                                    #If 'M' is signed and Trajectory Mode is on saved files will be executed in succession
                                    elif charac == 'M' and traj_mode == True:
                                         #Defines the string of combined letters
                                         sword = ''.join(word_spelled)
                                         print('Moving to selected saved files')
                                         
                                         #For every character in the string the arm will move to the saved angles of that file character
                                         for letter in sword:
                                             try:
                                                 if letter == 'G':
                                                     grip(close=True)
                                                     print('grip')
                                                     rospy.sleep(1)
                                                     
                                                 elif letter == 'U':
                                                     grip(close=False)
                                                     print('ungrip')
                                                     rospy.sleep(1)
                                                     
                                                 else:  
                                                     with open(f"saved_joint_angles_{letter}", "r") as f:
                                                         x = json.load(f)
                                                         limb.move_to_joint_positions(x, threshold=0.008726646, test=None)
                                                         
                                             #If a number, 'G', or 'U' is not signed, this error will be thrown
                                             except Exception as e:
                                                  word_spelled.pop()
                                                  print("Error: Letter not in directory")
                                                
                                            
                                         

 
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


        # Opens the camera feed
        cv2.imshow("Sawyer Head Camera - Gesture Recognition", frame)

        # Displays images for Joint Control Mode
        if joint_control_mode == True and change_increment == False and traj_mode == False:# and save_mode == False and execution_mode == False :
            # Defines the image created by the function
            display_img = create_display_image2(detected_gesture)
            # Defines the path where the image will be saved (tmp is used to store files temporarily)
            temp_image_path = "/tmp/sawyer_display.png"
            # Saves image to the path
            cv2.imwrite(temp_image_path, display_img)
            # Displays image to sawyer screen
            head_display.display_image(temp_image_path, display_in_loop=False, display_rate=10.0) 
        
        # Displays images for Change Increment
        elif change_increment == True: 
            display_img = create_display_image3(detected_gesture)
            temp_image_path = "/tmp/sawyer_display.png"
            cv2.imwrite(temp_image_path, display_img)  
            head_display.display_image(temp_image_path, display_in_loop=False, display_rate=10.0)
          
        # Displays images for Save Mode
        elif save_mode == True and pos_mode == False and joint_control_mode == False or execution_mode == True and pos_mode == False and joint_control_mode == False:
            display_img = create_display_image4(detected_gesture)
            temp_image_path = "/tmp/sawyer_display.png"
            cv2.imwrite(temp_image_path, display_img)  
            head_display.display_image(temp_image_path, display_in_loop=False, display_rate=10.0)

        # Displays images for Trajectory Mode
        elif traj_mode == True: 
            display_img = create_display_image5(detected_gesture, word_spelled)
            temp_image_path = "/tmp/sawyer_display.png"
            cv2.imwrite(temp_image_path, display_img)  
            head_display.display_image(temp_image_path, display_in_loop=False, display_rate=10.0)
       
        # Displays images for Position Mode
        else:
            display_img = create_display_image1(detected_gesture, word_spelled)
            temp_image_path = "/tmp/sawyer_display.png"
            cv2.imwrite(temp_image_path, display_img)  
            head_display.display_image(temp_image_path, display_in_loop=False, display_rate=10.0) 
              
        
        # Closes script if '.' is pressed
        if cv2.waitKey(1) & 0xFF == 46:
            #Displays the robot's default image
            image_path = '/home/ysc/Pictures/Default_Image.png'
            head_display.display_image(image_path, display_in_loop=False, display_rate=1.0)
            break
             
        elif charac == 'midfi' and execution_mode:
            image_path = '/home/ysc/Pictures/Default_Image.png'
            head_display.display_image(image_path, display_in_loop=False, display_rate=1.0)
            break
        
    # Closes all camera windows
    cv2.destroyAllWindows()

# Used as guard for against other scripts allowing it to run properly if called
if __name__ == "__main__":
    compare_gesture_live(threshold=0.07, hold_time=1.0, history_frames=5) 
