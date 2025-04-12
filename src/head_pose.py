from audioop import avg
from glob import glob
from itertools import count
import cv2
import mediapipe as mp
import numpy as np
import threading as th
import sounddevice as sd
import audio

# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 2
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

# Head pose thresholds
LEFT_THRESHOLD = -10
RIGHT_THRESHOLD = 10
DOWN_THRESHOLD = -10

# Global variables
x = 0  # X axis head pose
y = 0  # Y axis head pose
X_AXIS_CHEAT = 0
Y_AXIS_CHEAT = 0

def draw_head_pose_axes(image, nose_2d, rot_vec, trans_vec, cam_matrix, dist_matrix):
    """Draw 3D axes showing head orientation."""
    length = 50
    points = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length]])
    points_proj, _ = cv2.projectPoints(points, rot_vec, trans_vec, cam_matrix, dist_matrix)
    
    origin = (int(nose_2d[0]), int(nose_2d[1]))
    for i, (point, color) in enumerate(zip(points_proj, [BLUE, GREEN, RED])):
        point = (int(point[0][0]), int(point[0][1]))
        cv2.line(image, origin, point, color, 3)

def add_status_panel(image, x_angle, y_angle, looking_direction):
    """Add a status panel with head pose information."""
    h, w = image.shape[:2]
    # Draw semi-transparent overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
    
    # Add text
    cv2.putText(image, f"Head Direction: {looking_direction}", (20, 40), FONT, FONT_SCALE, WHITE, THICKNESS)
    cv2.putText(image, f"X Rotation: {int(x_angle)}°", (20, 70), FONT, FONT_SCALE, WHITE, THICKNESS)
    cv2.putText(image, f"Y Rotation: {int(y_angle)}°", (20, 100), FONT, FONT_SCALE, WHITE, THICKNESS)

def pose():
    global VOLUME_NORM, x, y, X_AXIS_CHEAT, Y_AXIS_CHEAT
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    
    # Smoothing filters
    x_filter = []
    y_filter = []
    filter_size = 5

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        
        face_ids = [33, 263, 1, 61, 291, 199]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh with custom color
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=GREEN, thickness=1, circle_radius=1))

                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in face_ids:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([
                    [focal_length, 0, img_h / 2],
                    [0, focal_length, img_w / 2],
                    [0, 0, 1]
                ])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360

                # Apply smoothing
                x_filter.append(x)
                y_filter.append(y)
                if len(x_filter) > filter_size:
                    x_filter.pop(0)
                    y_filter.pop(0)
                x = sum(x_filter) / len(x_filter)
                y = sum(y_filter) / len(y_filter)

                # Determine looking direction with smoother thresholds
                if y < LEFT_THRESHOLD:
                    looking_direction = "Looking Left"
                    X_AXIS_CHEAT = 1
                elif y > RIGHT_THRESHOLD:
                    looking_direction = "Looking Right"
                    X_AXIS_CHEAT = 1
                elif x < DOWN_THRESHOLD:
                    looking_direction = "Looking Down"
                    Y_AXIS_CHEAT = 1
                else:
                    looking_direction = "Forward"
                    X_AXIS_CHEAT = 0
                    Y_AXIS_CHEAT = 0

                # Draw visualization
                draw_head_pose_axes(image, nose_2d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                add_status_panel(image, x, y, looking_direction)

        # Add quit instructions
        cv2.putText(image, "Press 'ESC' to quit", (img_w - 200, 30), FONT, FONT_SCALE, WHITE, THICKNESS)
        
        cv2.imshow('Head Pose Estimation', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    t1 = th.Thread(target=pose)
    t1.start()
    t1.join()