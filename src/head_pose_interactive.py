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
FONT_SCALE = 0.6  
THICKNESS = 1  
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
GRAY = (128, 128, 128)
YELLOW = (0, 255, 255)

# UI Colors
PANEL_BG = (35, 35, 35)
ACCENT_COLOR = (255, 191, 0)  
WARNING_COLOR = (0, 69, 255) 
SUCCESS_COLOR = (0, 200, 83)  

# Head pose thresholds
LEFT_THRESHOLD = -10
RIGHT_THRESHOLD = 10
DOWN_THRESHOLD = -10

# Graph settings
GRAPH_HISTORY = 150  # Increased history for better trend visibility
UPDATE_INTERVAL = 33  # ~30 FPS update rate

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
        self.session_duration = 0
        
        # Initialize plots with modern style
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.patch.set_facecolor('#1C1C1C')
        self.setup_plots()
        
    def setup_plots(self):
        # Head movement plot
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_facecolor('#2C2C2C')
        self.ax1.set_title('Head Movement Analysis', color='white', pad=20, fontsize=12)
        self.ax1.set_ylabel('Angle (degrees)', color='white')
        self.ax1.grid(True, linestyle='--', alpha=0.3)
        self.line_x, = self.ax1.plot([], [], color='#FF6B6B', label='Vertical (Up/Down)', linewidth=2)
        self.line_y, = self.ax1.plot([], [], color='#4ECDC4', label='Horizontal (Left/Right)', linewidth=2)
        self.ax1.legend(facecolor='#2C2C2C', edgecolor='none')
        
        # Cheat detection plot
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_facecolor('#2C2C2C')
        self.ax2.set_title('Attention Monitoring', color='white', pad=20, fontsize=12)
        self.ax2.set_ylabel('Status', color='white')
        self.ax2.set_ylim(-0.5, 1.5)
        self.ax2.grid(True, linestyle='--', alpha=0.3)
        self.scatter_cheat = self.ax2.scatter([], [], c='#FF6B6B', marker='|', s=100)
        
        # Style improvements
        for ax in [self.ax1, self.ax2]:
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('#404040')
            ax.spines['top'].set_color('#404040')
            ax.spines['left'].set_color('#404040')
            ax.spines['right'].set_color('#404040')
        
        plt.tight_layout()

    def format_time(self, seconds):
        """Format time in MM:SS format"""
        return f"{int(seconds//60):02d}:{int(seconds%60):02d}"

    def draw_head_pose_axes(self, image, nose_2d, rot_vec, trans_vec, cam_matrix, dist_matrix):
        """Draw 3D axes showing head orientation."""
        length = 40  # Slightly shorter axes
        points = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length]])
        points_proj, _ = cv2.projectPoints(points, rot_vec, trans_vec, cam_matrix, dist_matrix)
        
        origin = (int(nose_2d[0]), int(nose_2d[1]))
        for i, (point, color) in enumerate(zip(points_proj, [BLUE, GREEN, RED])):
            point = (int(point[0][0]), int(point[0][1]))
            cv2.line(image, origin, point, color, 2)

    def add_status_panel(self, image, x_angle, y_angle, looking_direction):
        """Add a modern status panel with head pose information."""
        h, w = image.shape[:2]
        
        # Main panel background
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (300, 170), PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
        
        # Session time
        self.session_duration = time.time() - self.start_time
        time_str = self.format_time(self.session_duration)
        cv2.putText(image, f"Session Time: {time_str}", (20, 35), FONT, FONT_SCALE, ACCENT_COLOR, THICKNESS)
        
        # Direction indicator with dynamic color
        direction_color = WARNING_COLOR if X_AXIS_CHEAT or Y_AXIS_CHEAT else SUCCESS_COLOR
        cv2.putText(image, f"Direction: {looking_direction}", (20, 65), FONT, FONT_SCALE, direction_color, THICKNESS)
        
        # Rotation angles
        cv2.putText(image, f"Vertical: {int(x_angle)}°", (20, 95), FONT, FONT_SCALE, WHITE, THICKNESS)
        cv2.putText(image, f"Horizontal: {int(y_angle)}°", (20, 125), FONT, FONT_SCALE, WHITE, THICKNESS)
        
        # Status indicator
        status = "Attention Required" if X_AXIS_CHEAT or Y_AXIS_CHEAT else "Focused"
        status_color = WARNING_COLOR if X_AXIS_CHEAT or Y_AXIS_CHEAT else SUCCESS_COLOR
        cv2.putText(image, f"Status: {status}", (20, 155), FONT, FONT_SCALE, status_color, THICKNESS)
        
        # Add subtle border
        cv2.rectangle(image, (10, 10), (300, 170), GRAY, 1)

    def update_plot(self, frame):
        if len(time_data) > 0:
            # Update head movement plot with smooth lines
            self.line_x.set_data(time_data, [x[0] for x in x_data])
            self.line_y.set_data(time_data, [y[0] for y in y_data])
            
            # Update attention monitoring with improved visualization
            cheat_times = [t for t, c in zip(time_data, cheat_events) if c]
            cheat_values = [1 for _ in cheat_times]
            self.scatter_cheat.set_offsets(np.c_[cheat_times, cheat_values])
            
            # Dynamic axis updates
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.set_xlim(min(time_data), max(time_data))
            
            # Update labels
            self.ax2.set_xlabel(f'Time (s)', color='white')
            
        return self.line_x, self.line_y, self.scatter_cheat

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

            # Add minimal quit instructions
            cv2.putText(image, "ESC to exit", 
                       (image.shape[1] - 100, 30), FONT, FONT_SCALE, GRAY, THICKNESS)

            # Show the image
            cv2.imshow('Head Pose Analysis', image)

            # Check for exit
            if cv2.waitKey(5) & 0xFF == 27:
                self.running = False

        cap.release()
        cv2.destroyAllWindows()
        plt.close()

if __name__ == "__main__":
    monitor = HeadPoseMonitor()
    monitor.run() 