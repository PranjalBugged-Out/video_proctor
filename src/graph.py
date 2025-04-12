import cv2
import mediapipe as mp
import numpy as np
import threading as th
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import audio
from collections import deque
import datetime

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

# Graph settings
GRAPH_HISTORY = 100 
UPDATE_INTERVAL = 50 

# Global variables
x = 0  
y = 0  
X_AXIS_CHEAT = 0
Y_AXIS_CHEAT = 0

# Data storage for plotting
x_data = deque(maxlen=GRAPH_HISTORY)
y_data = deque(maxlen=GRAPH_HISTORY)
time_data = deque(maxlen=GRAPH_HISTORY)
cheat_events = deque(maxlen=GRAPH_HISTORY)

class HeadPoseMonitor:
    def __init__(self):
        self.running = True
        self.start_time = time.time()
        
        # Initialize plots
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(12, 8))
        self.setup_plots()
        
    def setup_plots(self):
        # Head movement plot
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title('Head Pose Angles Over Time')
        self.ax1.set_ylabel('Angle (degrees)')
        self.ax1.grid(True)
        self.line_x, = self.ax1.plot([], [], 'r-', label='X Rotation (Up/Down)')
        self.line_y, = self.ax1.plot([], [], 'b-', label='Y Rotation (Left/Right)')
        self.ax1.legend()
        
        # Cheat detection plot
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title('Cheat Detection Events')
        self.ax2.set_ylabel('Cheat Status')
        self.ax2.set_ylim(-0.5, 1.5)
        self.ax2.grid(True)
        self.scatter_cheat = self.ax2.scatter([], [], c='red', marker='x')
        
        plt.tight_layout()
    
    def update_plot(self, frame):
        if len(time_data) > 0:
            # Update head movement plot
            self.line_x.set_data(time_data, [x[0] for x in x_data])
            self.line_y.set_data(time_data, [y[0] for y in y_data])
            
            # Update cheat detection plot
            cheat_times = [t for t, c in zip(time_data, cheat_events) if c]
            cheat_values = [1 for _ in cheat_times]
            self.scatter_cheat.set_offsets(np.c_[cheat_times, cheat_values])
            
            # Update axis limits
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.set_xlim(min(time_data), max(time_data))
            
        return self.line_x, self.line_y, self.scatter_cheat

    def draw_head_pose_axes(self, image, nose_2d, rot_vec, trans_vec, cam_matrix, dist_matrix):
        """Draw 3D axes showing head orientation."""
        length = 50
        points = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length]])
        points_proj, _ = cv2.projectPoints(points, rot_vec, trans_vec, cam_matrix, dist_matrix)
        
        origin = (int(nose_2d[0]), int(nose_2d[1]))
        for i, (point, color) in enumerate(zip(points_proj, [BLUE, GREEN, RED])):
            point = (int(point[0][0]), int(point[0][1]))
            cv2.line(image, origin, point, color, 3)

    def add_status_panel(self, image, x_angle, y_angle, looking_direction):
        """Add a status panel with head pose information."""
        h, w = image.shape[:2]
        # Draw semi-transparent overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        
        # Add text
        cv2.putText(image, f"Head Direction: {looking_direction}", (20, 40), FONT, FONT_SCALE, WHITE, THICKNESS)
        cv2.putText(image, f"X Rotation: {int(x_angle)}°", (20, 70), FONT, FONT_SCALE, WHITE, THICKNESS)
        cv2.putText(image, f"Y Rotation: {int(y_angle)}°", (20, 100), FONT, FONT_SCALE, WHITE, THICKNESS)
        cv2.putText(image, f"Cheat Detection: {'Active' if X_AXIS_CHEAT or Y_AXIS_CHEAT else 'None'}", 
                    (20, 130), FONT, FONT_SCALE, RED if X_AXIS_CHEAT or Y_AXIS_CHEAT else GREEN, THICKNESS)

    def process_frame(self, face_mesh, image):
        global x, y, X_AXIS_CHEAT, Y_AXIS_CHEAT
        
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            self.process_landmarks(results.multi_face_landmarks[0], image)
            
        # Update data for plotting
        current_time = time.time() - self.start_time
        time_data.append(current_time)
        x_data.append((x, current_time))
        y_data.append((y, current_time))
        cheat_events.append(X_AXIS_CHEAT or Y_AXIS_CHEAT)
        
        return image

    def process_landmarks(self, face_landmarks, image):
        global x, y, X_AXIS_CHEAT, Y_AXIS_CHEAT
        
        img_h, img_w = image.shape[:2]
        face_3d = []
        face_2d = []
        face_ids = [33, 263, 1, 61, 291, 199]

        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in face_ids:
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                x_pos, y_pos = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x_pos, y_pos])
                face_3d.append([x_pos, y_pos, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # Camera matrix and pose estimation
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

        # Determine looking direction and update cheat detection
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
        self.draw_head_pose_axes(image, nose_2d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        self.add_status_panel(image, x, y, looking_direction)

    def run(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(0)

        # Create animation
        ani = FuncAnimation(self.fig, self.update_plot, interval=UPDATE_INTERVAL)
        plt.show(block=False)

        while self.running:
            success, image = cap.read()
            if not success:
                continue

            # Process frame
            image = self.process_frame(face_mesh, image)

            # Add quit instructions
            cv2.putText(image, "Press 'ESC' to quit", 
                       (image.shape[1] - 200, 30), FONT, FONT_SCALE, WHITE, THICKNESS)

            # Show the image
            cv2.imshow('Head Pose Estimation', image)

            # Check for exit
            if cv2.waitKey(5) & 0xFF == 27:
                self.running = False

        cap.release()
        cv2.destroyAllWindows()
        plt.close()

if __name__ == "__main__":
    monitor = HeadPoseMonitor()
    monitor.run()