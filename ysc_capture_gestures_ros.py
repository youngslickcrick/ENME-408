#! /usr/bin/env python3
# Chris Iheanacho @ UMBC
# Description: Captures gestures using Sawyer's head camera and MediaPipe Hand Tracking.
# ysc_capture_gestures_ros.py




# Section 1: Libary
# ------------------------------------------------------------------------------------------


#rospy: Python client library for ROS that enables programmers to communicate with ROS topics, services, and parameters. In other words it uses the ROS python API for communication with Sawyer.
#cv2(openCV): Imports the openCV library which provides functions for and classes for image processing and computer vision.
#os (operating systems): This allows for file manipulation and management, this is needed when saving images and javascript object notation (json) files to a specific directory.
#json(Javascript object notation): Used to store and transfer data.
#mediapipe: This is used for the hand tracking.
# Cvbridge: The bridge between ROS image messages and openCV image representation
# CvBridgeError: Used for handling errors in openCV

import rospy
import cv2
import os
import json 
import mediapipe as mp
import intera_interface
from cv_bridge import CvBridge, CvBridgeError



# Section 2: Global Variables, Mediapipe intialization and save directory
# ---------------------------------------------------------------------------------------------

# Variables that can be accessed in and out of a function
bridge = CvBridge()
latest_frame = None


#FROM MEDIAPIPE TEMPLATE: Mediapipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, 
                       max_num_hands=1,
                       min_detection_confidence=0.5, 
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

#Defining the directory where the images and coordinates of the gestures will be saved
SAVE_DIR = '/home/ysc/ros_ws/src/intera_sdk/intera_examples/scripts/captured_gestures_ros'

#Checks if directory exist, if not it will create it
os.makedirs(SAVE_DIR, exist_ok=True)

# Section 3: Basic Functions
# ---------------------------------------------------------------------------------------------
# Functions that will be accessed inside of the main function open_camera()

#Saves the image and the coordinates of the landmarks
def capture_gesture_image(frame, gesture_label, hand_landmarks):

    image_filename = f"{gesture_label}_asl_ros.jpg"
    json_filename = f"{gesture_label}_asl_landmarks_ros.json"
    
    #Joins the image filename to the path
    image_filepath = os.path.join(SAVE_DIR, image_filename)
    #Saves the camera feed to the image_filepath
    cv2.imwrite(image_filepath, frame)
    
    #Saves landmarks in a dictionary format
    landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]
    
    #Joints the coordinates filename to the path
    json_filepath = os.path.join(SAVE_DIR, json_filename)
    
    #Opens the json_filepath then writes the coordinates of the landmarks in json
    with open(json_filepath, 'w') as f:
        json.dump(landmarks, f)

    print(f"Captured gesture: {gesture_label} -> Saved as {image_filename} and {json_filename}")


#FROM RETHINKROBOTICS TEMPLATE: Processes images from Sawyer's head camera
def camera_callback(img_data, camera_name):
    #Using 'global' to call this variable
    global latest_frame
    
    #Try block allows one to test code for errors
    try:
        #Converting ros images messages to cv2 for processing
        latest_frame = bridge.imgmsg_to_cv2(img_data, "bgr8")
        #Mediapipe requries a flipped frame
        latest_frame = cv2.flip(latest_frame, 1)

    #Except handles the error  
    except CvBridgeError as e:
        rospy.logerr(f"CV Bridge Error: {e}")
        
# Section 4: Main Function
# ---------------------------------------------------------------------------------------------
# Opens the Sawyer head camera and captures hand gestures.
def open_camera():

    #Intializes rospy nodes for running
    rospy.init_node("sawyer_camera_capture", anonymous=True)

    # Defines cameras as sawyers Camera class
    cameras = intera_interface.Cameras()
    camera_name = "head_camera"
    
    # Verifies camera exists otherwise will throw an error
    if not cameras.verify_camera_exists(camera_name):
        rospy.logerr(f"Could not detect camera '{camera_name}', exiting.")
        return

    rospy.loginfo(f"Opening {camera_name}...")
    # Opens head camera
    cameras.start_streaming(camera_name)

    # Camera settings
    cameras.set_gain(camera_name, -1)
    cameras.set_exposure(camera_name, -1)

    #FROM RETHINKROBOTICS TEMPLATE: Callback method used to show the camera image
    cameras.set_callback(camera_name, camera_callback, rectify_image=True, callback_args=(camera_name,))

    print("Press 'a' for 'A', 'b' for 'B', 'c' for 'C', 'd' for 'D', etc. 'z' for 'Finish', 'j' for 'Backspace'. Press '.' to exit.")

    # Keeps the function looping as long ROS is active
    while not rospy.is_shutdown():
        if latest_frame is None:
            continue
        
        #Continously updates the frame
        frame = latest_frame.copy()
        
        #FROM MEDIAPIPE TEMPLATE: Used to convert frame to RGB which mediapipe runs in
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

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
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        #Opens the camera feed
        cv2.imshow("Sawyer Head Camera - Hand Tracking", frame)
        
        
        #Displays camera feed to sawyer head screen
        """
        head_display = intera_interface.HeadDisplay()
        temp_image_path = "/tmp/sawyer_head_camera_feed.jpg"
        cv2.imwrite(temp_image_path, frame)  
        head_display.display_image(temp_image_path, display_in_loop=False, display_rate=10.0)
        """
        
        key = cv2.waitKey(1) & 0xFF
        
        #If a key is pressed a letter is saved
        if key == ord('a'):
            capture_gesture_image(frame, 'A', hand_landmarks)  
        elif key == ord('b'):
            capture_gesture_image(frame, 'B', hand_landmarks)  
        elif key == ord('c'):
            capture_gesture_image(frame, 'C', hand_landmarks) 
        elif key == ord('d'):
            capture_gesture_image(frame, 'D', hand_landmarks)  
        elif key == ord('e'):
            capture_gesture_image(frame, 'E', hand_landmarks) 
        elif key == ord('f'):
            capture_gesture_image(frame, 'F', hand_landmarks)  
        elif key == ord('g'):
            capture_gesture_image(frame, 'G', hand_landmarks)  
        elif key == ord('h'):
            capture_gesture_image(frame, 'H', hand_landmarks)             
        elif key == ord('i'):
            capture_gesture_image(frame, 'I', hand_landmarks)             
        elif key == ord('k'):
            capture_gesture_image(frame, 'K', hand_landmarks)
        #elif key == ord('j'):
         #   capture_gesture_image(frame, 'J', hand_landmarks)   
        elif key == ord('l'):
            capture_gesture_image(frame, 'L', hand_landmarks) 
        elif key == ord('m'):
            capture_gesture_image(frame, 'M', hand_landmarks)            
        elif key == ord('n'):
            capture_gesture_image(frame, 'N', hand_landmarks)  
        elif key == ord('o'):
            capture_gesture_image(frame, 'O', hand_landmarks) 
        elif key == ord('p'):
            capture_gesture_image(frame, 'P', hand_landmarks)  
        elif key == ord('q'):
            capture_gesture_image(frame, 'Q', hand_landmarks)             
        elif key == ord('r'):
            capture_gesture_image(frame, 'R', hand_landmarks)               
        elif key == ord('s'):
            capture_gesture_image(frame, 'S', hand_landmarks)              
        elif key == ord('t'):
            capture_gesture_image(frame, 'T', hand_landmarks) 
        elif key == ord('u'):
            capture_gesture_image(frame, 'U', hand_landmarks)  
        elif key == ord('v'):
            capture_gesture_image(frame, 'V', hand_landmarks) 
        elif key == ord('w'):
            capture_gesture_image(frame, 'W', hand_landmarks)           
        elif key == ord('x'):
            capture_gesture_image(frame, 'X', hand_landmarks)                
        elif key == ord('y'):
            capture_gesture_image(frame, 'Y', hand_landmarks)
        elif key == ord('1'):
            capture_gesture_image(frame, '1', hand_landmarks)
        elif key == ord('2'):
            capture_gesture_image(frame, '2', hand_landmarks)  
        elif key == ord('3'):
            capture_gesture_image(frame, '3', hand_landmarks)
        elif key == ord('4'):
            capture_gesture_image(frame, '4', hand_landmarks)
        elif key == ord('5'):
            capture_gesture_image(frame, '5', hand_landmarks)  
        elif key == ord('6'):
            capture_gesture_image(frame, '6', hand_landmarks)            
        elif key == ord('7'):
            capture_gesture_image(frame, '7', hand_landmarks)
        elif key == ord('8'):
            capture_gesture_image(frame, '8', hand_landmarks)
        elif key == ord('9'):
            capture_gesture_image(frame, '9', hand_landmarks)                          
        elif key == ord('z'):
            capture_gesture_image(frame, 'finish', hand_landmarks)                
        elif key == ord('j'):
            capture_gesture_image(frame, 'backspace', hand_landmarks)
        #Closes script if '.' is pressed
        elif key == 46:  
            break
    
    cv2.destroyAllWindows()

#Used as guard for against other scripts allowing it to run properly if called
if __name__ == "__main__":
    open_camera()
