#! /usr/bin/env python3
# Chris Iheanacho @ UMBC
# Description: Captures gestures using Sawyer's head camera and MediaPipe Hand Tracking.
# ysc_capture_gestures_ros.py

import rospy
import cv2
import os
import json
import mediapipe as mp
import intera_interface
from cv_bridge import CvBridge, CvBridgeError

# Global variables
bridge = CvBridge()
latest_frame = None

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, 
                       max_num_hands=1,
                       min_detection_confidence=0.5, 
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Directory to save captured gestures
SAVE_DIR = '/home/ysc/ros_ws/src/intera_sdk/intera_examples/scripts/captured_gestures_ros'
os.makedirs(SAVE_DIR, exist_ok=True)

def capture_gesture_image(frame, gesture_label, hand_landmarks):
    """Capture the gesture image and save both image and landmarks as JSON."""
    image_filename = f"{gesture_label}_asl_ros.jpg"
    json_filename = f"{gesture_label}_asl_landmarks_ros.json"

    image_filepath = os.path.join(SAVE_DIR, image_filename)
    cv2.imwrite(image_filepath, frame)

    landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]
    
    json_filepath = os.path.join(SAVE_DIR, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(landmarks, f)

    print(f"Captured gesture: {gesture_label} -> Saved as {image_filename} and {json_filename}")

def camera_callback(img_data, camera_name):
    """Processes images from Sawyer's head camera and applies hand tracking."""
    global latest_frame
    try:
        # Convert ROS image to OpenCV format
        latest_frame = bridge.imgmsg_to_cv2(img_data, "bgr8")

        # Flip the frame horizontally
        latest_frame = cv2.flip(latest_frame, 1)

    except CvBridgeError as e:
        rospy.logerr(f"CV Bridge Error: {e}")

def open_camera():
    """Opens the Sawyer head camera and captures hand gestures."""
    rospy.init_node("sawyer_camera_capture", anonymous=True)

    # Initialize the Sawyer Camera API
    cameras = intera_interface.Cameras()
    camera_name = "head_camera"

    if not cameras.verify_camera_exists(camera_name):
        rospy.logerr(f"Could not detect camera '{camera_name}', exiting.")
        return

    rospy.loginfo(f"Opening {camera_name}...")
    cameras.start_streaming(camera_name)

    # Set gain and exposure (auto mode)
    cameras.set_gain(camera_name, -1)
    cameras.set_exposure(camera_name, -1)

    # Set callback function
    cameras.set_callback(camera_name, camera_callback, rectify_image=True, callback_args=(camera_name,))

    print("Press 'a' for 'A', 'b' for 'B', 'c' for 'C', 'd' for 'D', etc. '5' for 'Finish', '6' for 'Backspace'. Press 'ESC' to exit.")

    while not rospy.is_shutdown():
        if latest_frame is None:
            continue

        frame = latest_frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Define square box size (20% of frame height)
        box_size = int(height * 0.2)

        # Define new box position (higher in the frame)
        top_left = (width // 2 - box_size // 2, int(height * 0.25))  # Shifted up
        bottom_right = (top_left[0] + box_size, top_left[1] + box_size)

        # Draw green square box
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        cv2.imshow("Sawyer Head Camera - Hand Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            capture_gesture_image(frame, 'A', hand_landmarks)  # Capture and label 'A'
        elif key == ord('b'):
            capture_gesture_image(frame, 'B', hand_landmarks)  # Capture and label 'B'
        elif key == ord('c'):
            capture_gesture_image(frame, 'C', hand_landmarks)  # Capture and label 'C'
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
        elif key == ord('5'):
            capture_gesture_image(frame, 'finish', hand_landmarks)                
        elif key == ord('6'):
            capture_gesture_image(frame, 'backspace', hand_landmarks)
        elif key == 46:  # period key
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_camera()

