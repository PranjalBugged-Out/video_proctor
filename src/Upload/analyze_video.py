import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
from datetime import timedelta
import os

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Frame skipping settings for 5x speed
FRAME_SKIP = 5 

# Data storage for plotting and decision-making
time_data = []
eye_tracking_data = []
head_movement_data = []
mouth_movement_data = []
cheating_probability_data = []
mouth_movement_count = 0  # Count of detected talking/laughing events

class VideoAnalyzer:
    def __init__(self):
        self.video_path = None
        self.cheating_events = []
        self.frame_count = 0
        self.final_result = "Pending"
        self.is_analyzing = False
        self.current_frame = None
        self.current_probability = 0
        self.pause_analysis = False
        
    def reset_data(self):
        """Reset all data for new analysis."""
        global time_data, eye_tracking_data, head_movement_data
        global mouth_movement_data, cheating_probability_data, mouth_movement_count
        
        time_data = []
        eye_tracking_data = []
        head_movement_data = []
        mouth_movement_data = []
        cheating_probability_data = []
        mouth_movement_count = 0
        self.cheating_events = []
        self.frame_count = 0
        self.final_result = "Pending"
        self.current_probability = 0

    def analyze_video(self, video_path, frame_callback=None, progress_callback=None):
        """
        Analyze video with callbacks for UI updates.
        
        Args:
            video_path: Path to video file
            frame_callback: Function to call with processed frame
            progress_callback: Function to call with progress updates
        """
        global mouth_movement_count
        self.reset_data()
        self.is_analyzing = True

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file.")
            self.is_analyzing = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip_count = 0

        while cap.isOpened() and self.is_analyzing:
            if self.pause_analysis:
                time.sleep(0.1)  # Reduce CPU usage while paused
                continue
                
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            frame_skip_count += 1

            # Skip frames to speed up processing (4x speed)
            if frame_skip_count < FRAME_SKIP:
                continue
            frame_skip_count = 0

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect facial landmarks
            results = mp_face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    eye_tracking = self.get_eye_tracking(face_landmarks, frame.shape)
                    head_movement = self.get_head_movement(face_landmarks, frame.shape)
                    mouth_movement = self.get_mouth_movement(face_landmarks)

                    # Store movement data
                    elapsed_time = self.frame_count / fps
                    time_data.append(elapsed_time)
                    eye_tracking_data.append(eye_tracking)
                    head_movement_data.append(head_movement)
                    mouth_movement_data.append(mouth_movement)

                    # Count mouth movements (talking/laughing)
                    if mouth_movement > 5:  # Threshold for talking/laughing
                        mouth_movement_count += 1

                    # Calculate cheating probability
                    cheating_probability = self.calculate_cheating_probability(eye_tracking, head_movement, mouth_movement)
                    cheating_probability_data.append(cheating_probability)
                    self.current_probability = cheating_probability

                    # Detect high cheating probability
                    if cheating_probability > 60:
                        self.log_cheating_event(self.frame_count, f"Cheating Probability: {cheating_probability:.2f}%")
                    
                    # Draw facial landmarks and metrics on frame
                    self.draw_metrics_on_frame(frame, eye_tracking, head_movement, mouth_movement, cheating_probability)
            
            # Store current frame for UI
            self.current_frame = frame
            
            # Call frame callback if provided
            if frame_callback:
                frame_callback(frame)
                
            # Update progress
            if progress_callback and total_frames > 0:
                progress = (self.frame_count / total_frames) * 100
                progress_callback(progress)
                
            # Slow down playback to match 4x speed
            time.sleep(0.01)  # Adjust for smoother playback

        cap.release()
        self.generate_final_result()
        self.is_analyzing = False

    def draw_metrics_on_frame(self, frame, eye_tracking, head_movement, mouth_movement, cheating_probability):
        """Draw metrics on the video frame."""
        h, w = frame.shape[:2]
        
        # Enhanced constants for better text rendering
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        TITLE_FONT_SCALE = 0.75
        FONT_SCALE = 0.65
        THICKNESS = 2
        THIN_THICKNESS = 1
        
        # Enhanced colors for better visibility
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        CYAN = (255, 255, 0)  # Yellow in BGR
        GREEN = (0, 255, 0)
        MAGENTA = (255, 0, 255)
        BLUE = (255, 0, 0)
        RED = (0, 0, 255)
        ORANGE = (0, 165, 255)
        PANEL_BG = (25, 25, 25)
        
        # Create a semi-transparent overlay for the metrics panel
        overlay = frame.copy()
        panel_width = 320
        panel_height = 150
        panel_x = 15
        panel_y = 15
        
        # Draw panel background with rounded corners effect
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     PANEL_BG, -1)
        
        # Add a border for better definition
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     WHITE, 1)
        
        # Apply the overlay with transparency
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add panel title
        cv2.putText(frame, "ANALYSIS METRICS", 
                   (panel_x + 10, panel_y + 25), 
                   FONT, TITLE_FONT_SCALE, WHITE, THICKNESS, cv2.LINE_AA)
        
        # Add horizontal separator
        cv2.line(frame, (panel_x + 10, panel_y + 35), 
                (panel_x + panel_width - 10, panel_y + 35), 
                WHITE, 1)
        
        # Add metrics text with improved rendering
        # Eye movement
        metric_y = panel_y + 60
        label = "Eye Movement:"
        value = f"{eye_tracking:.1f}"
        cv2.putText(frame, label, (panel_x + 15, metric_y), 
                   FONT, FONT_SCALE, WHITE, THIN_THICKNESS, cv2.LINE_AA)
        
        # Draw value with background highlight for better visibility
        text_size = cv2.getTextSize(value, FONT, FONT_SCALE, THICKNESS)[0]
        value_x = panel_x + 170
        
        # Value background
        cv2.rectangle(frame, 
                     (value_x - 5, metric_y - text_size[1] - 5), 
                     (value_x + text_size[0] + 5, metric_y + 5), 
                     (50, 50, 50), -1)
        
        cv2.putText(frame, value, (value_x, metric_y), 
                   FONT, FONT_SCALE, CYAN, THICKNESS, cv2.LINE_AA)
        
        # Head movement
        metric_y = panel_y + 85
        label = "Head Movement:"
        value = f"{head_movement:.1f}"
        cv2.putText(frame, label, (panel_x + 15, metric_y), 
                   FONT, FONT_SCALE, WHITE, THIN_THICKNESS, cv2.LINE_AA)
        
        # Value background
        text_size = cv2.getTextSize(value, FONT, FONT_SCALE, THICKNESS)[0]
        cv2.rectangle(frame, 
                     (value_x - 5, metric_y - text_size[1] - 5), 
                     (value_x + text_size[0] + 5, metric_y + 5), 
                     (50, 50, 50), -1)
        
        cv2.putText(frame, value, (value_x, metric_y), 
                   FONT, FONT_SCALE, GREEN, THICKNESS, cv2.LINE_AA)
        
        # Mouth movement
        metric_y = panel_y + 110
        label = "Mouth Movement:"
        value = f"{mouth_movement:.1f}"
        cv2.putText(frame, label, (panel_x + 15, metric_y), 
                   FONT, FONT_SCALE, WHITE, THIN_THICKNESS, cv2.LINE_AA)
        
        # Value background
        text_size = cv2.getTextSize(value, FONT, FONT_SCALE, THICKNESS)[0]
        cv2.rectangle(frame, 
                     (value_x - 5, metric_y - text_size[1] - 5), 
                     (value_x + text_size[0] + 5, metric_y + 5), 
                     (50, 50, 50), -1)
        
        cv2.putText(frame, value, (value_x, metric_y), 
                   FONT, FONT_SCALE, MAGENTA, THICKNESS, cv2.LINE_AA)
        
        # Cheating probability with enhanced color coding and visualization
        metric_y = panel_y + 140
        label = "Cheating Probability:"
        value = f"{cheating_probability:.1f}%"
        
        # Determine color based on probability
        if cheating_probability > 60:
            color = RED  # High risk
        elif cheating_probability > 30:
            color = ORANGE  # Medium risk
        else:
            color = GREEN  # Low risk
        
        cv2.putText(frame, label, (panel_x + 15, metric_y), 
                   FONT, FONT_SCALE, WHITE, THIN_THICKNESS, cv2.LINE_AA)
        
        # Create a more prominent background for the probability value
        text_size = cv2.getTextSize(value, FONT, FONT_SCALE + 0.1, THICKNESS)[0]
        value_x = panel_x + 170
        
        # Draw a background that matches the risk level
        cv2.rectangle(frame, 
                     (value_x - 5, metric_y - text_size[1] - 5), 
                     (value_x + text_size[0] + 5, metric_y + 5), 
                     (50, 50, 50), -1)
        
        # Add the probability value with enhanced visibility
        cv2.putText(frame, value, (value_x, metric_y), 
                   FONT, FONT_SCALE + 0.1, color, THICKNESS, cv2.LINE_AA)
        
        # Add talking/laughing counter with improved visibility
        if mouth_movement_count > 0:
            # Create a separate panel for talking events
            event_panel_width = 200
            event_panel_height = 50
            event_panel_x = w - event_panel_width - 15
            event_panel_y = 15
            
            # Draw panel with transparency
            overlay = frame.copy()
            cv2.rectangle(overlay, (event_panel_x, event_panel_y), 
                         (event_panel_x + event_panel_width, event_panel_y + event_panel_height), 
                         PANEL_BG, -1)
            
            # Add border
            cv2.rectangle(overlay, (event_panel_x, event_panel_y), 
                         (event_panel_x + event_panel_width, event_panel_y + event_panel_height), 
                         MAGENTA, 1)
            
            # Apply transparency
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Add title and count
            cv2.putText(frame, "TALKING EVENTS", 
                       (event_panel_x + 10, event_panel_y + 20), 
                       FONT, FONT_SCALE, MAGENTA, THIN_THICKNESS, cv2.LINE_AA)
            
            # Add count with background highlight
            count_text = f"{mouth_movement_count}"
            text_size = cv2.getTextSize(count_text, FONT, TITLE_FONT_SCALE, THICKNESS)[0]
            count_x = event_panel_x + event_panel_width - text_size[0] - 15
            
            # Background for count
            cv2.rectangle(frame, 
                         (count_x - 5, event_panel_y + 25 - text_size[1] - 5), 
                         (count_x + text_size[0] + 5, event_panel_y + 25 + 5), 
                         (50, 50, 50), -1)
            
            cv2.putText(frame, count_text, 
                       (count_x, event_panel_y + 25), 
                       FONT, TITLE_FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

    def get_eye_tracking(self, face_landmarks, shape):
        """Calculate eye tracking movement intensity."""
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]

        return abs(left_eye.x - right_eye.x) * 100  # Normalize value

    def get_head_movement(self, face_landmarks, shape):
        """Calculate head movement intensity."""
        nose = face_landmarks.landmark[1]

        return abs(nose.x - 0.5) * 200  # Normalize around center

    def get_mouth_movement(self, face_landmarks):
        """Detect mouth movement based on lip position."""
        upper_lip = face_landmarks.landmark[13].y
        lower_lip = face_landmarks.landmark[14].y

        return abs(upper_lip - lower_lip) * 100  # Normalize value

    def calculate_cheating_probability(self, eye_tracking, head_movement, mouth_movement):
        """Determine the probability of cheating based on movement data."""
        cheating_score = (0.4 * eye_tracking) + (0.4 * head_movement) + (0.2 * mouth_movement)
        return min(100, cheating_score)  # Cap at 100%

    def log_cheating_event(self, frame_num, event_type):
        """Log cheating events."""
        timestamp = round(frame_num / 30, 2)
        self.cheating_events.append(f"Time: {timestamp}s - {event_type}")

    def generate_final_result(self):
        """Determine final selection based on cheating probability."""
        if len(cheating_probability_data) == 0:
            self.final_result = "No data available"
            return
            
        if mouth_movement_count > 5:
            self.final_result = "Rejected (Excessive Talking)"
            return

        avg_cheating_probability = sum(cheating_probability_data) / len(cheating_probability_data)
        if avg_cheating_probability > 50:
            self.final_result = "Rejected (Suspicious Behavior)"
        else:
            self.final_result = "Selected (No Suspicious Behavior)"

class EnhancedVideoUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Video Proctoring System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        self.analyzer = VideoAnalyzer()
        self.setup_ui()
        
        # Animation update
        self.update_id = None
        
    def setup_ui(self):
        """Set up the enhanced UI with integrated video and graphs."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Upload button
        self.upload_btn = ttk.Button(control_frame, text="Upload Video", command=self.upload_video)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Analyze button
        self.analyze_btn = ttk.Button(control_frame, text="Analyze", command=self.analyze_video, state="disabled")
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Pause/Resume button
        self.pause_btn = ttk.Button(control_frame, text="Pause", command=self.toggle_pause, state="disabled")
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        # Stop button
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_analysis, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # File info label
        self.file_label = ttk.Label(control_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=20)
        
        # Content pane (video + graphs)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Video frame (left side)
        video_frame = ttk.LabelFrame(content_frame, text="Video Analysis")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Video canvas
        self.video_canvas = tk.Canvas(video_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Graphs frame (right side)
        graphs_frame = ttk.LabelFrame(content_frame, text="Real-time Metrics")
        graphs_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create matplotlib figure for graphs
        self.fig = Figure(figsize=(6, 8), dpi=100)
        self.fig.patch.set_facecolor('#f0f0f0')
        
        # Movement graph
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title("Movement Tracking")
        self.ax1.set_xlabel("Time (seconds)")
        self.ax1.set_ylabel("Movement Intensity")
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Cheating probability graph
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("Cheating Probability")
        self.ax2.set_xlabel("Time (seconds)")
        self.ax2.set_ylabel("Probability (%)")
        self.ax2.set_ylim(0, 100)
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add horizontal reference lines for probability
        self.ax2.axhline(y=30, color='green', linestyle='-', alpha=0.3)
        self.ax2.axhline(y=60, color='red', linestyle='-', alpha=0.3)
        
        # Add canvas to display the figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=graphs_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bottom status panel
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(status_frame, variable=self.progress_var, length=100)
        self.progress.pack(fill=tk.X, side=tk.TOP)
        
        # Status label
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Result label
        self.result_label = ttk.Label(status_frame, text="", font=("Arial", 10, "bold"))
        self.result_label.pack(side=tk.RIGHT, padx=5)
        
        # Configure figure for better spacing
        self.fig.tight_layout()
    
    def upload_video(self):
        """Open file dialog to upload video."""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if file_path:
            self.analyzer.video_path = file_path
            self.file_label.config(text=f"File: {os.path.basename(file_path)}")
            self.analyze_btn.config(state="normal")
            self.status_label.config(text="Video loaded. Ready to analyze.")
    
    def analyze_video(self):
        """Start video analysis in a separate thread."""
        if not self.analyzer.video_path:
            messagebox.showerror("Error", "No video file selected!")
            return
        
        # Reset UI
        self.progress_var.set(0)
        self.status_label.config(text="Analyzing video...")
        self.result_label.config(text="")
        self.analyze_btn.config(state="disabled")
        self.pause_btn.config(state="normal")
        self.stop_btn.config(state="normal")
        
        # Start analysis thread
        threading.Thread(
            target=self.analyzer.analyze_video, 
            args=(self.analyzer.video_path, self.update_video_frame, self.update_progress),
            daemon=True
        ).start()
        
        # Start animation update
        self.start_animation()
    
    def toggle_pause(self):
        """Pause or resume analysis."""
        if self.analyzer.is_analyzing:
            if self.analyzer.pause_analysis:
                self.analyzer.pause_analysis = False
                self.pause_btn.config(text="Pause")
                self.status_label.config(text="Analysis resumed")
            else:
                self.analyzer.pause_analysis = True
                self.pause_btn.config(text="Resume")
                self.status_label.config(text="Analysis paused")
    
    def stop_analysis(self):
        """Stop the analysis."""
        if self.analyzer.is_analyzing:
            self.analyzer.is_analyzing = False
            self.status_label.config(text="Analysis stopped")
            self.pause_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            self.analyze_btn.config(state="normal")
            
            # Stop animation
            if self.update_id:
                self.root.after_cancel(self.update_id)
    
    def update_video_frame(self, frame):
        """Update the video display with the current frame."""
        if frame is not None:
            # Resize frame to fit canvas
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has size
                # Calculate aspect ratio
                frame_height, frame_width = frame.shape[:2]
                aspect_ratio = frame_width / frame_height
                
                # Determine new dimensions
                if canvas_width / canvas_height > aspect_ratio:
                    # Canvas is wider than frame
                    new_height = canvas_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    # Canvas is taller than frame
                    new_width = canvas_width
                    new_height = int(new_width / aspect_ratio)
                
                # Resize frame
                frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = tk.PhotoImage(data=bytes(cv2.imencode('.ppm', frame_rgb)[1].tobytes()))
                
                # Update canvas
                self.video_canvas.create_image(
                    canvas_width // 2, canvas_height // 2, 
                    image=img, anchor=tk.CENTER
                )
                self.video_canvas.image = img  # Keep reference
    
    def update_progress(self, progress):
        """Update progress bar."""
        self.progress_var.set(progress)
        
        # Update status
        if progress >= 100:
            self.status_label.config(text="Analysis complete")
            self.result_label.config(text=f"Result: {self.analyzer.final_result}")
            self.pause_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            self.analyze_btn.config(state="normal")
    
    def start_animation(self):
        """Start the animation loop for updating graphs."""
        self.update_graphs()
    
    def update_graphs(self):
        """Update the graphs with current data."""
        if not self.analyzer.is_analyzing and len(time_data) == 0:
            # No data to display
            self.update_id = self.root.after(100, self.update_graphs)
            return
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Set titles and labels
        self.ax1.set_title("Movement Tracking")
        self.ax1.set_xlabel("Time (seconds)")
        self.ax1.set_ylabel("Movement Intensity")
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        
        self.ax2.set_title("Cheating Probability")
        self.ax2.set_xlabel("Time (seconds)")
        self.ax2.set_ylabel("Probability (%)")
        self.ax2.set_ylim(0, 100)
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add reference lines
        self.ax2.axhline(y=30, color='green', linestyle='-', alpha=0.3)
        self.ax2.axhline(y=60, color='red', linestyle='-', alpha=0.3)
        
        if len(time_data) > 0:
            # Plot movement data
            self.ax1.plot(time_data, eye_tracking_data, label="Eye Movement", color='cyan')
            self.ax1.plot(time_data, head_movement_data, label="Head Movement", color='lime')
            self.ax1.plot(time_data, mouth_movement_data, label="Mouth Movement", color='magenta')
            self.ax1.legend(loc='upper left')
            
            # Plot cheating probability
            self.ax2.plot(time_data, cheating_probability_data, label="Cheating Probability", 
                         color='red', linewidth=2)
            
            # Add color bands for probability levels
            self.ax2.axhspan(0, 30, alpha=0.2, color='green')
            self.ax2.axhspan(30, 60, alpha=0.2, color='yellow')
            self.ax2.axhspan(60, 100, alpha=0.2, color='red')
            
            # Add current probability indicator
            if len(time_data) > 0:
                self.ax2.scatter([time_data[-1]], [self.analyzer.current_probability], 
                               color='blue', s=100, zorder=5)
        
        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Schedule next update
        self.update_id = self.root.after(100, self.update_graphs)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedVideoUI(root)
    root.mainloop()

