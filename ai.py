
# import cv2
# import mediapipe as mp
# import numpy as np
# import tkinter as tk

# # Create a function to calculate the angle between three points
# def calculate_angle(a, b, c):
#     a = np.array(a)  # Shoulder
#     b = np. array(b)  # Elbow
#     c = np.array(c)  # Wrist
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(np.degrees(radians))
#     return angle

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# # VIDEO FEED
# cap = cv2.VideoCapture(0)

# # Curl counter variables
# counter = 0
# stage = "down"  # Initialize as "down"
# prev_angle = None

# # Function to show an alert and close it after a delay
# def show_alert():
#     alert = tk.Tk()
#     alert.title("Incorrect Bicep Curl")
#     label = tk.Label(alert, text="Incorrect Bicep Curl!").pack()
#     alert.after(2000, alert.destroy)  # Close the alert window after 2 seconds (2000 milliseconds)
#     alert.mainloop()

# # Define valid angle range for correct curl
# min_valid_angle = 40  # Adjust this value as needed
# max_valid_angle = 180  # Adjust this value as needed

# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()

#         # Recolor image to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False

#         # Make detection
#         results = pose.process(image)

#         # Recolor back to BGR
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # Extract landmarks
#         try:
#             landmarks = results.pose_landmarks.landmark

#             shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#             elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#             wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

#             angle = calculate_angle(shoulder, elbow, wrist)

#             if prev_angle is None:
#                 prev_angle = angle

#             cv2.putText(image, str(int(angle)),
#                         (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#             # Curl counter logic
#             if stage == "down" and angle > 160:
#                 stage = "up"
#                 print("Up stage angle:", angle)
#             if stage == "up" and (angle < min_valid_angle or angle > max_valid_angle):
#                 stage = "down"
#                 show_alert()  # Show an alert for incorrect curl
#                 print("Incorrect curl angle:", angle)
#             if stage == "up" and min_valid_angle <= angle <= max_valid_angle:
#                 # This is a correct curl within the valid range
#                 pass

#             prev_angle = angle

#         except:
#             pass

#         cv2.putText(image, 'REPS', (10, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#         cv2.putText(image, str(counter),
#                     (10, 150),
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#                                   )

#         cv2.imshow('Mediapipe Feed', image)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np

# Create a function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # Shoulder
    b = np.array(b)  # Elbow
    c = np.array(c)  # Wrist
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# VIDEO FEED
cap = cv2.VideoCapture(0)

# Curl counter variables
counter = 0
stage = "down"  # Initialize as "down"
prev_angle = None
up_stage_angle = None  # Initialize up stage angle
down_stage_angle = None  # Initialize down stage angle

# Define valid angle range for correct curl
min_valid_angle = 70  # Adjust this value as needed
max_valid_angle = 170  # Adjust this value as needed

# Define criteria for incorrect bicep curl
max_up_angle = 45  # Maximum angle for the "up" stage
min_down_angle = 150  # Minimum angle for the "down" stage

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            angle = calculate_angle(shoulder, elbow, wrist)

            if prev_angle is None:
                prev_angle = angle

            cv2.putText(image, f"Angle: {int(angle)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)  # Black text

            # Curl counter logic
            if stage == "down" and angle > max_valid_angle:
                stage = "up"
                up_stage_angle = angle  # Record up stage angle
                print("Up stage angle:", angle)
            if stage == "up" and (angle < max_up_angle or angle > min_down_angle):
                stage = "down"
                down_stage_angle = angle  # Record down stage angle
                print("Down stage angle:", angle)
            if stage == "up" and min_valid_angle <= angle <= max_valid_angle:
                # This is a correct curl within the valid range
                pass

            prev_angle = angle

        except:
            pass

        cv2.putText(image, 'REPS', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)  # Black text
        cv2.putText(image, str(counter),
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)  # Black text

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()