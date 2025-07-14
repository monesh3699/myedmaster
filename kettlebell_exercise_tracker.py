#!/usr/bin/env python3
"""
Kettlebell Pull Exercise Tracker

A comprehensive computer vision-based system for tracking kettlebell pull exercises 
with real-time feedback, rep counting, and detailed workout logs.

Features:
- Live camera or video file input
- Real-time rep counting and form analysis
- Interactive workout guidance
- Comprehensive session logging
- Progress tracking and analytics

Exercise Focus:
- Kettlebell High Pull
- Kettlebell Upright Row
- Single-arm Kettlebell Row

Author: AI Assistant
Created: 2025
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

import cv2
import mediapipe as mp
import numpy as np
import math
import json
import datetime
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("✅ All dependencies imported successfully")

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

class ExerciseConfig:
    """Configuration constants for kettlebell exercises"""
    
   
    
    SINGLE_ARM_ROW = {
        'elbow_angle_up': 160,         # Arm extended
        'elbow_angle_down': 80,        # Arm contracted
        'shoulder_angle_up': 20,       # Shoulder neutral
        'shoulder_angle_down': 45      # Shoulder pulled back
    }
    
    # MediaPipe confidence thresholds
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.7
    
    # UI Colors (BGR format for OpenCV)
    COLORS = {
        'primary': (0, 255, 0),      # Green
        'secondary': (255, 0, 0),    # Blue
        'accent': (0, 255, 255),     # Yellow
        'warning': (0, 165, 255),    # Orange
        'error': (0, 0, 255),        # Red
        'text': (255, 255, 255)      # White
    }
    
    # Logging settings
    LOG_DIRECTORY = "workout_logs"
    SESSION_DATA_FILE = "session_data.json"

print("✅ Configuration loaded successfully")

# =============================================================================
# COMPREHENSIVE LOGGING SYSTEM
# =============================================================================

class WorkoutLogger:
    """Comprehensive logging system for workout sessions"""
    
    def __init__(self, log_dir: str = ExerciseConfig.LOG_DIRECTORY):
        self.log_dir = log_dir
        self.session_id = None
        self.session_data = {}
        self.frame_logs = []
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def start_session(self, exercise_type: str, input_source: str) -> str:
        """Start a new workout session"""
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.session_data = {
            'session_id': self.session_id,
            'exercise_type': exercise_type,
            'input_source': input_source,
            'start_time': datetime.datetime.now().isoformat(),
            'end_time': None,
            'duration_seconds': 0,
            'total_reps': 0,
            'rep_timestamps': [],
            'form_feedback': [],
            'angles_history': [],
            'stage_transitions': [],
            'session_notes': ""
        }
        
        self.frame_logs = []
        return self.session_id
    
    def log_frame(self, frame_number: int, angles: Dict, stage: str, feedback: List[str]):
        """Log data for a single frame"""
        frame_data = {
            'frame': frame_number,
            'timestamp': datetime.datetime.now().isoformat(),
            'angles': angles,
            'stage': stage,
            'feedback': feedback
        }
        
        self.frame_logs.append(frame_data)
        self.session_data['angles_history'].append(angles)
    
    def log_rep(self, rep_count: int, stage_from: str, stage_to: str):
        """Log a completed repetition"""
        rep_data = {
            'rep_number': rep_count,
            'timestamp': datetime.datetime.now().isoformat(),
            'stage_transition': f"{stage_from} -> {stage_to}"
        }
        
        self.session_data['rep_timestamps'].append(rep_data)
        self.session_data['stage_transitions'].append(rep_data)
        self.session_data['total_reps'] = rep_count
    
    def log_feedback(self, feedback: str, feedback_type: str = "form"):
        """Log form feedback"""
        feedback_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'type': feedback_type,
            'message': feedback
        }
        
        self.session_data['form_feedback'].append(feedback_data)
    
    def end_session(self, notes: str = "") -> str:
        """End the current session and save logs"""
        if not self.session_id:
            return None
        
        end_time = datetime.datetime.now()
        start_time = datetime.datetime.fromisoformat(self.session_data['start_time'])
        duration = (end_time - start_time).total_seconds()
        
        self.session_data['end_time'] = end_time.isoformat()
        self.session_data['duration_seconds'] = duration
        self.session_data['session_notes'] = notes
        
        # Save session data
        session_file = os.path.join(self.log_dir, f"session_{self.session_id}.json")
        with open(session_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        # Save detailed frame logs
        frames_file = os.path.join(self.log_dir, f"frames_{self.session_id}.json")
        with open(frames_file, 'w') as f:
            json.dump(self.frame_logs, f, indent=2)
        
        return session_file
    
    def get_session_summary(self) -> Dict:
        """Get current session summary"""
        if not self.session_data:
            return {}
        
        duration = self.session_data.get('duration_seconds', 0)
        total_reps = self.session_data.get('total_reps', 0)
        
        return {
            'session_id': self.session_id,
            'exercise_type': self.session_data.get('exercise_type', 'Unknown'),
            'duration_minutes': round(duration / 60, 2),
            'total_reps': total_reps,
            'reps_per_minute': round((total_reps / duration * 60), 1) if duration > 0 else 0,
            'feedback_count': len(self.session_data.get('form_feedback', []))
        }
    
    @staticmethod
    def load_session(session_file: str) -> Dict:
        """Load a previous session from file"""
        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    @staticmethod
    def list_sessions(log_dir: str = None) -> List[str]:
        """List all available session files"""
        if log_dir is None:
            log_dir = ExerciseConfig.LOG_DIRECTORY
        
        if not os.path.exists(log_dir):
            return []
        
        session_files = [f for f in os.listdir(log_dir) if f.startswith('session_') and f.endswith('.json')]
        return sorted(session_files, reverse=True)  # Most recent first

print("✅ Logging system initialized")

# =============================================================================
# INTERACTIVE UI COMPONENTS
# =============================================================================

class WorkoutUI:
    """Interactive UI components for workout feedback"""
    
    def __init__(self):
        self.current_instruction = ""
        self.motivation_messages = [
            "Keep it up!",
            "Perfect form!",
            "You're crushing it!",
            "Smooth movement!",
            "Great control!",
            "Focus on the pull!",
            "Engage your core!",
            "Nice rhythm!"
        ]
        self.instruction_queue = []
    
    def draw_workout_hud(self, frame: np.ndarray, rep_count: int, stage: str, 
                        exercise_type: str, feedback: List[str], 
                        session_time: float) -> np.ndarray:
        """Draw comprehensive heads-up display on the workout frame"""
        height, width = frame.shape[:2]
        
        # Main info panel (top-left)
        self._draw_info_panel(frame, rep_count, stage, exercise_type, session_time)
        
        # Instructions panel (top-right)
        self._draw_instructions_panel(frame, width)
        
        # Feedback panel (bottom-left)
        self._draw_feedback_panel(frame, feedback, height)
        
        # Progress indicator (bottom)
        self._draw_progress_bar(frame, rep_count, width, height)
        
        return frame
    
    def _draw_info_panel(self, frame: np.ndarray, rep_count: int, stage: str, 
                        exercise_type: str, session_time: float):
        """Draw main information panel"""
        # Background panel
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), ExerciseConfig.COLORS['primary'], 2)
        
        # Exercise type
        cv2.putText(frame, exercise_type.upper(), (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, ExerciseConfig.COLORS['accent'], 2)
        
        # Rep counter (large)
        cv2.putText(frame, f"REPS: {rep_count:02d}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, ExerciseConfig.COLORS['primary'], 3)
        
        # Current stage
        stage_color = ExerciseConfig.COLORS['secondary'] if stage == 'up' else ExerciseConfig.COLORS['warning']
        cv2.putText(frame, f"Stage: {stage.upper()}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, stage_color, 2)
        
        # Session time
        minutes = int(session_time // 60)
        seconds = int(session_time % 60)
        cv2.putText(frame, f"Time: {minutes:02d}:{seconds:02d}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ExerciseConfig.COLORS['text'], 1)
    
    def _draw_instructions_panel(self, frame: np.ndarray, width: int):
        """Draw exercise instructions panel"""
        panel_width = 250
        panel_x = width - panel_width - 10
        
        # Background
        cv2.rectangle(frame, (panel_x, 10), (width - 10, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, 10), (width - 10, 100), ExerciseConfig.COLORS['secondary'], 2)
        
        # Title
        cv2.putText(frame, "NEXT ACTION:", (panel_x + 10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ExerciseConfig.COLORS['accent'], 2)
        
        # Current instruction
        instruction = self.get_current_instruction()
        self._draw_wrapped_text(frame, instruction, panel_x + 10, 55, 
                               panel_width - 20, ExerciseConfig.COLORS['text'])
    
    def _draw_feedback_panel(self, frame: np.ndarray, feedback: List[str], height: int):
        """Draw form feedback panel"""
        if not feedback:
            return
        
        panel_height = min(len(feedback) * 25 + 40, 150)
        panel_y = height - panel_height - 50
        
        # Background
        cv2.rectangle(frame, (10, panel_y), (350, height - 50), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, panel_y), (350, height - 50), ExerciseConfig.COLORS['accent'], 2)
        
        # Title
        cv2.putText(frame, "FORM FEEDBACK:", (20, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ExerciseConfig.COLORS['accent'], 2)
        
        # Feedback messages
        for i, msg in enumerate(feedback[-5:]):  # Show last 5 messages
            y_pos = panel_y + 50 + (i * 20)
            color = ExerciseConfig.COLORS['primary'] if "Good" in msg or "Perfect" in msg else ExerciseConfig.COLORS['warning']
            cv2.putText(frame, f"• {msg}", (25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_progress_bar(self, frame: np.ndarray, rep_count: int, width: int, height: int):
        """Draw workout progress bar"""
        # Target reps for session (can be made configurable)
        target_reps = 15
        progress = min(rep_count / target_reps, 1.0)
        
        bar_width = width - 40
        bar_height = 20
        bar_x = 20
        bar_y = height - 30
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     ExerciseConfig.COLORS['text'], 1)
        
        # Progress fill
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         ExerciseConfig.COLORS['primary'], -1)
        
        # Progress text
        progress_text = f"{rep_count}/{target_reps} reps ({int(progress*100)}%)"
        text_x = bar_x + bar_width // 2 - 60
        cv2.putText(frame, progress_text, (text_x, bar_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ExerciseConfig.COLORS['text'], 1)
    
    def _draw_wrapped_text(self, frame: np.ndarray, text: str, x: int, y: int, 
                          max_width: int, color: Tuple[int, int, int]):
        """Draw text with word wrapping"""
        words = text.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            if text_size[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x, y + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def update_instruction(self, stage: str, exercise_type: str, form_quality: str):
        """Update workout instruction based on current state"""
        if exercise_type == "KETTLEBELL_HIGH_PULL":
            if stage == "down":
                self.current_instruction = "Pull the kettlebell up explosively! Drive with your hips."
            else:
                self.current_instruction = "Control the descent. Keep your core engaged."
        elif exercise_type == "KETTLEBELL_UPRIGHT_ROW":
            if stage == "down":
                self.current_instruction = "Lift elbows high, keep kettlebell close to body."
            else:
                self.current_instruction = "Lower with control, feel the stretch."
        else:  # Single arm row
            if stage == "down":
                self.current_instruction = "Pull elbow back, squeeze shoulder blade."
            else:
                self.current_instruction = "Extend arm fully, maintain posture."
        
        # Add form-specific guidance
        if form_quality == "poor":
            self.current_instruction += " Focus on form over speed!"
    
    def get_current_instruction(self) -> str:
        """Get the current workout instruction"""
        return self.current_instruction if self.current_instruction else "Position yourself and begin the exercise"
    
    def get_motivation_message(self, rep_count: int) -> str:
        """Get a motivational message based on progress"""
        if rep_count > 0 and rep_count % 5 == 0:
            return np.random.choice(self.motivation_messages)
        return ""

print("✅ UI components ready")

# =============================================================================
# KETTLEBELL EXERCISE ANALYSIS ENGINE
# =============================================================================

class KettlebellExerciseAnalyzer:
    """Core analysis engine for kettlebell pull exercises"""
    
    def __init__(self, exercise_type: str = "SINGLE_ARM_ROW"):
        self.exercise_type = exercise_type
        self.config = getattr(ExerciseConfig, exercise_type)
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=ExerciseConfig.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=ExerciseConfig.MIN_TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Exercise tracking state
        self.rep_count = 0
        self.stage = "down"
        self.last_stage = "down"
        self.form_feedback = []
        self.angle_history = []
        
    def calculate_angle(self, point1: List[float], point2: List[float], point3: List[float]) -> float:
        """Calculate angle between three points using vectors"""
        a = np.array(point1)
        b = np.array(point2)  # Vertex point
        c = np.array(point3)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def get_landmark_coordinates(self, landmarks, landmark_id: int) -> List[float]:
        """Extract normalized coordinates from landmark"""
        landmark = landmarks[landmark_id]
        return [landmark.x, landmark.y]
    
    def analyze_kettlebell_high_pull(self, landmarks) -> Tuple[Dict[str, float], List[str]]:
        """Analyze kettlebell high pull exercise form"""
        # Get key landmarks for both sides
        left_shoulder = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        left_elbow = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW.value)
        left_wrist = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST.value)
        left_hip = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP.value)
        
        right_shoulder = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        right_elbow = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        right_wrist = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST.value)
        right_hip = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        
        # Calculate angles
        left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_shoulder_angle = self.calculate_angle(left_hip, left_shoulder, left_elbow)
        right_shoulder_angle = self.calculate_angle(right_hip, right_shoulder, right_elbow)
        
        # Use dominant side (or average both sides)
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        avg_shoulder_angle = (left_shoulder_angle + right_shoulder_angle) / 2
        
        angles = {
            'elbow_angle': avg_elbow_angle,
            'shoulder_angle': avg_shoulder_angle,
            'left_elbow': left_elbow_angle,
            'right_elbow': right_elbow_angle
        }
        
        # Form analysis
        feedback = []
        
        # Check pull height
        if avg_shoulder_angle < self.config['shoulder_angle_down']:
            feedback.append("Great pull height!")
        elif avg_shoulder_angle > 120:
            feedback.append("Pull higher - drive with hips!")
        
        # Check elbow position
        if avg_elbow_angle < self.config['elbow_angle_down']:
            feedback.append("Good elbow drive!")
        elif avg_elbow_angle > 150:
            feedback.append("Bend elbows more at top")
        
        # Check symmetry
        elbow_diff = abs(left_elbow_angle - right_elbow_angle)
        if elbow_diff > 20:
            feedback.append("Keep both arms symmetric")
        else:
            feedback.append("Good bilateral control")
        
        return angles, feedback
    
    def analyze_upright_row(self, landmarks) -> Tuple[Dict[str, float], List[str]]:
        """Analyze kettlebell upright row exercise form"""
        # Get key landmarks
        left_shoulder = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        left_elbow = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW.value)
        left_wrist = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST.value)
        
        right_shoulder = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        right_elbow = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        right_wrist = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST.value)
        
        # Calculate angles
        left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Check elbow height relative to shoulders
        left_elbow_height = left_shoulder[1] - left_elbow[1]  # Negative means elbow above shoulder
        right_elbow_height = right_shoulder[1] - right_elbow[1]
        
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        avg_elbow_height = (left_elbow_height + right_elbow_height) / 2
        
        angles = {
            'elbow_angle': avg_elbow_angle,
            'elbow_height': avg_elbow_height,
            'left_elbow': left_elbow_angle,
            'right_elbow': right_elbow_angle
        }
        
        # Form analysis
        feedback = []
        
        if avg_elbow_height < -0.05:  # Elbows well above shoulders
            feedback.append("Perfect elbow height!")
        elif avg_elbow_height > 0:
            feedback.append("Lift elbows higher than shoulders")
        
        if avg_elbow_angle < self.config['elbow_angle_down']:
            feedback.append("Good elbow bend at top!")
        
        # Check if kettlebell stays close to body (wrists should be close to center)
        wrist_distance = abs(left_wrist[0] - right_wrist[0])
        if wrist_distance < 0.3:  # Normalized coordinates
            feedback.append("Keep kettlebell close to body")
        else:
            feedback.append("Bring kettlebell closer to body")
        
        return angles, feedback
    
    def analyze_single_arm_row(self, landmarks, dominant_side: str = "right") -> Tuple[Dict[str, float], List[str]]:
        """Analyze single-arm kettlebell row exercise form"""
        # Select landmarks based on dominant side
        if dominant_side == "right":
            shoulder = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            elbow = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            wrist = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST.value)
            hip = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        else:
            shoulder = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            elbow = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW.value)
            wrist = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST.value)
            hip = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP.value)
        
        # Calculate angles
        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        shoulder_angle = self.calculate_angle(hip, shoulder, elbow)
        
        angles = {
            'elbow_angle': elbow_angle,
            'shoulder_angle': shoulder_angle
        }
        
        # Form analysis
        feedback = []
        
        if elbow_angle < self.config['elbow_angle_down']:
            feedback.append("Perfect elbow pull!")
        elif elbow_angle > 150:
            feedback.append("Pull elbow back more")
        
        # Check if elbow is pulled behind shoulder line (good rowing form)
        if (dominant_side == "right" and elbow[0] < shoulder[0]) or \
           (dominant_side == "left" and elbow[0] > shoulder[0]):
            feedback.append("Great rowing form!")
        else:
            feedback.append("Pull elbow past shoulder line")
        
        if shoulder_angle > self.config['shoulder_angle_down']:
            feedback.append("Good shoulder blade squeeze")
        
        return angles, feedback
    
    def analyze_form(self, landmarks) -> Tuple[Dict[str, float], List[str]]:
        """Main form analysis dispatcher"""
        if self.exercise_type == "KETTLEBELL_HIGH_PULL":
            return self.analyze_kettlebell_high_pull(landmarks)
        elif self.exercise_type == "KETTLEBELL_UPRIGHT_ROW":
            return self.analyze_upright_row(landmarks)
        elif self.exercise_type == "SINGLE_ARM_ROW":
            return self.analyze_single_arm_row(landmarks)
        else:
            return {}, ["Unknown exercise type"]
    
    def count_repetitions(self, angles: Dict[str, float]) -> bool:
        """Count repetitions based on exercise-specific logic"""
        rep_completed = False
        previous_stage = self.stage
        
        if self.exercise_type in ["KETTLEBELL_HIGH_PULL", "KETTLEBELL_UPRIGHT_ROW"]:
            # Use shoulder angle for high pull and upright row
            angle = angles.get('shoulder_angle', angles.get('elbow_angle', 180))
            
            # Debug: Print angle and thresholds every 30 frames
            if hasattr(self, 'debug_counter'):
                self.debug_counter += 1
            else:
                self.debug_counter = 1
            
            if self.debug_counter % 30 == 0:
                print(f"DEBUG: Angle={angle:.1f}, Up_threshold={self.config['shoulder_angle_up']}, Down_threshold={self.config['shoulder_angle_down']}, Stage={self.stage}")
            
            # Stage transitions based on angle thresholds
            if angle > self.config['shoulder_angle_up']:
                self.stage = "down"
            elif angle < self.config['shoulder_angle_down']:
                self.stage = "up"
            
            # Count rep on transition from down to up
            if previous_stage == "down" and self.stage == "up":
                self.rep_count += 1
                rep_completed = True
                self.last_stage = previous_stage
                print(f"Rep completed! Total: {self.rep_count}")
        
        elif self.exercise_type == "SINGLE_ARM_ROW":
            # Use elbow angle for single arm row
            elbow_angle = angles.get('elbow_angle', 180)
            
            # Stage transitions based on angle thresholds
            if elbow_angle > self.config['elbow_angle_up']:
                self.stage = "down"
            elif elbow_angle < self.config['elbow_angle_down']:
                self.stage = "up"
            
            # Count rep on transition from down to up
            if previous_stage == "down" and self.stage == "up":
                self.rep_count += 1
                rep_completed = True
                self.last_stage = previous_stage
                print(f"Rep completed! Total: {self.rep_count}")
        
        return rep_completed
    
    def get_form_quality(self, feedback: List[str]) -> str:
        """Assess overall form quality based on feedback"""
        positive_keywords = ['Perfect', 'Great', 'Good', 'control', 'height', 'drive', 'squeeze']
        positive_count = sum(1 for fb in feedback if any(kw in fb for kw in positive_keywords))
        
        if positive_count >= len(feedback) * 0.7:  # 70% positive feedback
            return "excellent"
        elif positive_count >= len(feedback) * 0.4:  # 40% positive feedback
            return "good"
        else:
            return "poor"
    
    def reset_counter(self):
        """Reset repetition counter and stage"""
        self.rep_count = 0
        self.stage = "down"
        self.form_feedback = []
        self.angle_history = []

print("✅ Exercise analysis engine ready")

# =============================================================================
# MAIN KETTLEBELL TRACKER APPLICATION
# =============================================================================

class KettlebellWorkoutTracker:
    """Main application class combining all components"""
    
    def __init__(self, exercise_type: str = "SINGLE_ARM_ROW"):
        # Initialize components
        self.exercise_analyzer = KettlebellExerciseAnalyzer(exercise_type)
        self.logger = WorkoutLogger()
        self.ui = WorkoutUI()
        
        # Session state
        self.session_active = False
        self.session_start_time = None
        self.frame_count = 0
        self.input_source = None
        
        print(f"✅ Single-Arm Kettlebell Row Tracker initialized")
    
    def start_session(self, input_source: str) -> str:
        """Start a new workout session"""
        if self.session_active:
            print("⚠️ Session already active. End current session first.")
            return None
        
        self.session_start_time = datetime.datetime.now()
        self.session_active = True
        self.input_source = input_source
        self.frame_count = 0
        
        # Reset analyzer state
        self.exercise_analyzer.reset_counter()
        
        # Start logging
        session_id = self.logger.start_session(self.exercise_analyzer.exercise_type, input_source)
        
        print(f"🎯 Session started: {session_id}")
        print(f"📹 Input source: {input_source}")
        print(f"🏋️ Exercise: {self.exercise_analyzer.exercise_type}")
        
        return session_id
    
    def end_session(self, notes: str = "") -> str:
        """End the current workout session"""
        if not self.session_active:
            print("⚠️ No active session to end.")
            return None
        
        self.session_active = False
        
        # End logging and save data
        log_file = self.logger.end_session(notes)
        
        # Print session summary
        summary = self.logger.get_session_summary()
        print("\n" + "="*50)
        print("🏁 WORKOUT SESSION COMPLETE")
        print("="*50)
        print(f"📊 Total Reps: {summary['total_reps']}")
        print(f"⏱️ Duration: {summary['duration_minutes']} minutes")
        print(f"🚀 Reps/Minute: {summary['reps_per_minute']}")
        print(f"💬 Feedback Count: {summary['feedback_count']}")
        print(f"💾 Session saved: {log_file}")
        print("="*50)
        
        return log_file
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single video frame"""
        if not self.session_active:
            return frame
        
        self.frame_count += 1
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Pose detection
        results = self.exercise_analyzer.pose.process(rgb_frame)
        
        # Convert back to BGR
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Process pose landmarks if detected
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Analyze exercise form
            angles, feedback = self.exercise_analyzer.analyze_form(landmarks)
            
            # Count repetitions
            rep_completed = self.exercise_analyzer.count_repetitions(angles)
            
            # Log rep completion
            if rep_completed:
                self.logger.log_rep(
                    self.exercise_analyzer.rep_count,
                    self.exercise_analyzer.last_stage,
                    self.exercise_analyzer.stage
                )
                
                # Log motivational feedback
                motivation = self.ui.get_motivation_message(self.exercise_analyzer.rep_count)
                if motivation:
                    self.logger.log_feedback(motivation, "motivation")
            
            # Update UI instructions
            form_quality = self.exercise_analyzer.get_form_quality(feedback)
            self.ui.update_instruction(
                self.exercise_analyzer.stage,
                self.exercise_analyzer.exercise_type,
                form_quality
            )
            
            # Log frame data every 10 frames to reduce overhead
            if self.frame_count % 10 == 0:
                self.logger.log_frame(
                    self.frame_count,
                    angles,
                    self.exercise_analyzer.stage,
                    feedback
                )
            
            # Log feedback
            for fb in feedback:
                if fb not in self.exercise_analyzer.form_feedback[-3:]:  # Avoid duplicate logging
                    self.logger.log_feedback(fb, "form")
            
            self.exercise_analyzer.form_feedback = feedback
            
            # Draw pose landmarks
            self.exercise_analyzer.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.exercise_analyzer.mp_pose.POSE_CONNECTIONS,
                self.exercise_analyzer.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                self.exercise_analyzer.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            
            # Draw workout HUD
            session_time = (datetime.datetime.now() - self.session_start_time).total_seconds()
            frame = self.ui.draw_workout_hud(
                frame,
                self.exercise_analyzer.rep_count,
                self.exercise_analyzer.stage,
                self.exercise_analyzer.exercise_type,
                feedback,
                session_time
            )
        
        else:
            # No pose detected - show instruction
            cv2.putText(frame, "Position yourself in camera view", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, ExerciseConfig.COLORS['warning'], 2)
            cv2.putText(frame, "Stand sideways for best detection", (50, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, ExerciseConfig.COLORS['text'], 2)
        
        return frame
    
    def run_webcam_session(self):
        """Run workout session with webcam input"""
        print("\n🔄 Initializing webcam...")
        
        # Try different camera backends for macOS compatibility
        cap = None
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY, 0]
        
        for backend in backends:
            if backend == 0:
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(0, backend)
            
            if cap is not None and cap.isOpened():
                print(f"✅ Camera opened with backend: {backend}")
                break
            else:
                if cap is not None:
                    cap.release()
                print(f"⚠️ Failed with backend: {backend}")
        
        if cap is None or not cap.isOpened():
            print("❌ Error: Could not access webcam with any backend")
            print("\n🔧 TROUBLESHOOTING TIPS:")
            print("1. Check if another app is using the camera (Zoom, Teams, etc.)")
            print("2. Grant camera permissions to Terminal/Python in System Preferences")
            print("3. Try restarting the application")
            print("4. Use a video file instead (option 2 in main menu)")
            return
        
        # Set camera properties for better compatibility
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test camera by reading a few frames
        print("🔍 Testing camera connection...")
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera test {i+1}/5 successful")
                break
            else:
                print(f"⚠️ Camera test {i+1}/5 failed, retrying...")
                import time
                time.sleep(0.5)
        
        if not ret:
            print("❌ Error: Camera is not providing frames")
            print("\n🔧 CAMERA TROUBLESHOOTING:")
            print("1. Camera may need a moment to initialize - try again")
            print("2. Check camera permissions in System Preferences > Security & Privacy")
            print("3. Close other apps that might be using the camera")
            print("4. Try unplugging and reconnecting external cameras")
            cap.release()
            return
        
        # Start session
        session_id = self.start_session("webcam")
        
        print("\n" + "="*60)
        print("🏋️ SINGLE-ARM KETTLEBELL ROW SESSION STARTED")
        print("="*60)
        print("📋 Controls:")
        print("   • Press 'q' to quit and end session")
        print("   • Press 'r' to reset rep counter")
        print("   • Press 'p' to pause/resume")
        print("   • Press 's' to take screenshot")
        print("📏 Position yourself sideways to the camera for best tracking")
        print("🎯 Ready to track your single-arm kettlebell rows!")
        print("="*60)
        print("🎬 Camera feed should appear in a new window...")
        
        paused = False
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                frame_count += 1
                
                if not ret:
                    print(f"⚠️ Failed to read frame {frame_count}")
                    print("🔄 Attempting to reconnect...")
                    cap.release()
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        print("❌ Could not reconnect to camera")
                        break
                    continue
                
                if not paused:
                    # Process frame
                    frame = self.process_frame(frame)
                else:
                    # Show pause overlay
                    cv2.putText(frame, "PAUSED - Press 'p' to resume", 
                               (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, ExerciseConfig.COLORS['warning'], 3)
                
                # Display frame
                cv2.imshow('Single-Arm Kettlebell Row Tracker', frame)
                
                # Handle key presses
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.exercise_analyzer.reset_counter()
                    print("🔄 Rep counter reset!")
                elif key == ord('p'):
                    paused = not paused
                    print(f"⏸️ Session {'paused' if paused else 'resumed'}")
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = f"screenshot_{self.logger.session_id}_{self.frame_count}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Session interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            print("💡 This might be a camera access issue")
        
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            if cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            
            # Only ask for notes if session was actually started
            if self.session_active:
                try:
                    notes = input("\n📝 Add session notes (optional): ").strip()
                except (EOFError, KeyboardInterrupt):
                    notes = "Session ended abruptly"
                self.end_session(notes)
            else:
                print("⚠️ Session was not properly initialized")
    
    def run_video_session(self, video_path: str, output_path: str = None, show_video: bool = True):
        """Run workout session with video file input"""
        print(f"\n🎬 Loading video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not load video file {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"📊 Video Info:")
        print(f"   • Resolution: {width}x{height}")
        print(f"   • FPS: {fps}")
        print(f"   • Duration: {duration:.2f} seconds")
        print(f"   • Total Frames: {total_frames}")
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Output will be saved to: {output_path}")
        
        # Start session
        session_id = self.start_session(f"video: {video_path}")
        
        print("\n" + "="*60)
        print("🏋️ SINGLE-ARM KETTLEBELL ROW VIDEO ANALYSIS STARTED")
        print("="*60)
        if show_video:
            print("📋 Controls:")
            print("   • Press 'q' to quit early")
            print("   • Press 'p' to pause/resume")
            print("   • Press SPACE for next frame (when paused)")
        print("🎯 Analyzing single-arm kettlebell rows...")
        print("="*60)
        
        frame_count = 0
        paused = False
        
        try:
            while cap.isOpened():
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Process frame
                    processed_frame = self.process_frame(frame)
                    
                    # Add video progress indicator
                    progress = frame_count / total_frames
                    progress_text = f"Progress: {progress*100:.1f}% | Frame: {frame_count}/{total_frames}"
                    cv2.putText(processed_frame, progress_text, (10, height - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, ExerciseConfig.COLORS['text'], 1)
                    
                    # Save frame if output specified
                    if out:
                        out.write(processed_frame)
                    
                    # Display frame if requested
                    if show_video:
                        cv2.imshow('Single-Arm Kettlebell Row Analysis', processed_frame)
                        
                        # Control playback speed
                        key = cv2.waitKey(max(1, int(1000/fps))) & 0xFF
                        if key == ord('q'):
                            print("⏹️ Analysis stopped by user")
                            break
                        elif key == ord('p'):
                            paused = True
                            print("⏸️ Analysis paused. Press 'p' to resume or SPACE for next frame.")
                    
                    # Print progress every 5% or every 10 seconds
                    if frame_count % max(1, total_frames // 20) == 0:
                        reps = self.exercise_analyzer.rep_count
                        print(f"📈 Progress: {progress*100:.1f}% | Reps: {reps}")
                
                else:  # Paused
                    if show_video:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('p'):
                            paused = False
                            print("▶️ Analysis resumed")
                        elif key == ord(' '):  # Space for single frame
                            ret, frame = cap.read()
                            if ret:
                                frame_count += 1
                                processed_frame = self.process_frame(frame)
                                cv2.imshow('Single-Arm Kettlebell Row Analysis', processed_frame)
                        elif key == ord('q'):
                            break
                    else:
                        break
        
        except KeyboardInterrupt:
            print("\n⏹️ Analysis interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if show_video:
                cv2.destroyAllWindows()
            
            # End session
            notes = f"Video analysis of {video_path}. Processed {frame_count}/{total_frames} frames."
            self.end_session(notes)

print("✅ Main tracker application ready")

# =============================================================================
# SESSION MANAGEMENT AND ANALYTICS
# =============================================================================

class SessionAnalytics:
    """Analytics and visualization for workout sessions"""
    
    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir if log_dir else ExerciseConfig.LOG_DIRECTORY
    
    def load_session_data(self, session_file: str) -> Dict:
        """Load session data from file"""
        try:
            session_path = os.path.join(self.log_dir, session_file)
            with open(session_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Session file not found: {session_file}")
            return {}
        except json.JSONDecodeError:
            print(f"❌ Invalid session file format: {session_file}")
            return {}
    
    def analyze_session_performance(self, session_data: Dict) -> Dict:
        """Analyze session performance metrics"""
        if not session_data:
            return {}
        
        analysis = {
            'session_id': session_data.get('session_id', 'Unknown'),
            'exercise_type': session_data.get('exercise_type', 'Unknown'),
            'total_reps': session_data.get('total_reps', 0),
            'duration_minutes': round(session_data.get('duration_seconds', 0) / 60, 2),
            'reps_per_minute': 0,
            'consistency_score': 0,
            'form_quality_score': 0,
            'improvement_areas': []
        }
        
        duration = session_data.get('duration_seconds', 0)
        if duration > 0:
            analysis['reps_per_minute'] = round((analysis['total_reps'] / duration) * 60, 1)
        
        # Analyze rep timestamps for consistency
        rep_timestamps = session_data.get('rep_timestamps', [])
        if len(rep_timestamps) > 1:
            intervals = []
            for i in range(1, len(rep_timestamps)):
                prev_time = datetime.datetime.fromisoformat(rep_timestamps[i-1]['timestamp'])
                curr_time = datetime.datetime.fromisoformat(rep_timestamps[i]['timestamp'])
                interval = (curr_time - prev_time).total_seconds()
                intervals.append(interval)
            
            if intervals:
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                # Lower coefficient of variation means better consistency
                cv = std_interval / mean_interval if mean_interval > 0 else 1
                analysis['consistency_score'] = max(0, 100 - (cv * 100))
        
        # Analyze form feedback
        form_feedback = session_data.get('form_feedback', [])
        if form_feedback:
            positive_keywords = ['Perfect', 'Great', 'Good', 'control', 'height', 'drive', 'squeeze']
            total_feedback = len(form_feedback)
            positive_feedback = sum(1 for fb in form_feedback 
                                  if any(kw in fb.get('message', '') for kw in positive_keywords))
            
            analysis['form_quality_score'] = round((positive_feedback / total_feedback) * 100, 1)
            
            # Identify improvement areas
            common_issues = {}
            for fb in form_feedback:
                message = fb.get('message', '')
                if any(word in message.lower() for word in ['more', 'better', 'improve', 'higher']):
                    for word in message.split():
                        if word.lower() in ['elbow', 'shoulder', 'form', 'height', 'pull']:
                            common_issues[word.lower()] = common_issues.get(word.lower(), 0) + 1
            
            analysis['improvement_areas'] = sorted(common_issues.items(), 
                                                 key=lambda x: x[1], reverse=True)[:3]
        
        return analysis
    
    def generate_session_report(self, session_file: str) -> str:
        """Generate comprehensive session report"""
        session_data = self.load_session_data(session_file)
        if not session_data:
            return "❌ Unable to load session data"
        
        analysis = self.analyze_session_performance(session_data)
        
        report = f"""
{'='*80}
📊 KETTLEBELL WORKOUT SESSION REPORT
{'='*80}

🎯 SESSION DETAILS:
   • Session ID: {analysis['session_id']}
   • Exercise Type: {analysis['exercise_type']}
   • Date: {session_data.get('start_time', 'Unknown')[:10]}
   • Duration: {analysis['duration_minutes']} minutes
   • Input Source: {session_data.get('input_source', 'Unknown')}

🏋️ PERFORMANCE METRICS:
   • Total Repetitions: {analysis['total_reps']}
   • Reps per Minute: {analysis['reps_per_minute']}
   • Consistency Score: {analysis['consistency_score']:.1f}/100
   • Form Quality Score: {analysis['form_quality_score']:.1f}/100

📈 PERFORMANCE ASSESSMENT:
"""
        
        # Performance assessment
        if analysis['form_quality_score'] >= 80:
            report += "   ✅ Excellent form quality - keep up the great work!\n"
        elif analysis['form_quality_score'] >= 60:
            report += "   👍 Good form quality - minor improvements needed\n"
        else:
            report += "   ⚠️ Form needs attention - focus on technique\n"
        
        if analysis['consistency_score'] >= 70:
            report += "   ✅ Great consistency in rep timing\n"
        elif analysis['consistency_score'] >= 50:
            report += "   👍 Moderate consistency - try to maintain steady rhythm\n"
        else:
            report += "   ⚠️ Inconsistent timing - focus on steady pace\n"
        
        # Improvement areas
        if analysis['improvement_areas']:
            report += "\n🎯 AREAS FOR IMPROVEMENT:\n"
            for area, count in analysis['improvement_areas']:
                report += f"   • {area.title()}: mentioned {count} times\n"
        
        # Session notes
        if session_data.get('session_notes'):
            report += f"\n📝 SESSION NOTES:\n   {session_data['session_notes']}\n"
        
        report += "\n" + "="*80
        
        return report
    
    def list_all_sessions(self) -> List[Dict]:
        """List all available sessions with summary info"""
        session_files = WorkoutLogger.list_sessions(self.log_dir)
        sessions = []
        
        for session_file in session_files:
            session_data = self.load_session_data(session_file)
            if session_data:
                analysis = self.analyze_session_performance(session_data)
                sessions.append({
                    'file': session_file,
                    'id': analysis['session_id'],
                    'exercise': analysis['exercise_type'],
                    'date': session_data.get('start_time', 'Unknown')[:10],
                    'reps': analysis['total_reps'],
                    'duration': analysis['duration_minutes'],
                    'form_score': analysis['form_quality_score']
                })
        
        return sessions

print("✅ Session analytics ready")

# =============================================================================
# INPUT SOURCE SELECTION AND MAIN INTERFACE
# =============================================================================

def display_welcome_screen():
    """Display welcome screen with instructions"""
    welcome_text = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                   🏋️ SINGLE-ARM KETTLEBELL ROW TRACKER 🏋️                  ║
    ║                                                                              ║
    ║  A comprehensive computer vision system for tracking single-arm KB rows     ║
    ║                                                                              ║
    ║  📊 FEATURES:                                                                ║
    ║    • Real-time rep counting and form analysis                               ║
    ║    • Interactive workout guidance                                           ║
    ║    • Comprehensive session logging                                          ║
    ║    • Progress tracking and analytics                                        ║
    ║    • Camera or video file input                                             ║
    ║                                                                              ║
    ║  🎯 EXERCISE FOCUS:                                                          ║
    ║    • Single-Arm Kettlebell Row                                              ║
    ║    • Elbow pull-back tracking                                               ║
    ║    • Shoulder blade engagement                                              ║
    ║    • Proper rowing form analysis                                            ║
    ║                                                                              ║
    ║  💡 TIPS FOR BEST RESULTS:                                                  ║
    ║    • Position yourself sideways to the camera                              ║
    ║    • Ensure good lighting and clear background                             ║
    ║    • Wear contrasting clothing for better detection                        ║
    ║    • Keep your whole body in frame                                         ║
    ║    • Use your dominant arm for best tracking                               ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    
    print(welcome_text)

def get_user_exercise_choice():
    """Get exercise type from user - Single-Arm Row only"""
    print("\n🎯 EXERCISE TYPE: Single-Arm Kettlebell Row")
    print("   Focus: Pull elbow back, squeeze shoulder blade")
    return 'SINGLE_ARM_ROW'

def get_user_input_choice():
    """Get input source from user"""
    print("\n📹 SELECT INPUT SOURCE:")
    print("1. Live Camera/Webcam 📹")
    print("2. Video File 🎬")
    
    while True:
        choice = input("\nEnter your choice (1-2): ").strip()
        if choice == '1':
            return 'camera'
        elif choice == '2':
            return 'video'
        print("❌ Invalid choice. Please enter 1 or 2.")

def get_video_file_path():
    """Get video file path from user"""
    while True:
        video_path = input("\n📁 Enter video file path: ").strip()
        if os.path.exists(video_path):
            return video_path
        print(f"❌ File not found: {video_path}")
        print("💡 Please check the path and try again.")

def get_output_settings():
    """Get output settings for video processing"""
    save_output = input("\n💾 Save analyzed video? (y/n): ").strip().lower() == 'y'
    
    if save_output:
        output_path = input("📁 Enter output path (or press Enter for default): ").strip()
        if not output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"analyzed_workout_{timestamp}.mp4"
        return output_path
    
    return None

def view_session_history():
    """View workout session history"""
    analytics = SessionAnalytics()
    sessions = analytics.list_all_sessions()
    
    if not sessions:
        print("📭 No workout sessions found.")
        print("💡 Start your first workout to begin tracking!")
        return
    
    print("\n" + "="*100)
    print("📊 WORKOUT SESSION HISTORY")
    print("="*100)
    print(f"{'Date':<12} {'Session ID':<10} {'Exercise':<25} {'Reps':<6} {'Duration':<10} {'Form Score':<12}")
    print("-"*100)
    
    for session in sessions:
        print(f"{session['date']:<12} {session['id'][-8:]:<10} {session['exercise']:<25} "
              f"{session['reps']:<6} {session['duration']:<10.1f} {session['form_score']:<12.1f}%")
    
    print("="*100)
    print(f"📈 Total Sessions: {len(sessions)}")
    print(f"🏋️ Total Reps: {sum(s['reps'] for s in sessions)}")
    print(f"⏱️ Total Time: {sum(s['duration'] for s in sessions):.1f} minutes")

def view_session_analytics():
    """View detailed session analytics"""
    analytics = SessionAnalytics()
    sessions = analytics.list_all_sessions()
    
    if not sessions:
        print("📭 No workout sessions found for analysis.")
        print("💡 Complete some workouts first to view analytics!")
        return
    
    print("\n📈 SESSION ANALYTICS")
    print("="*50)
    
    # Show latest session report
    latest_session = sessions[0]['file']
    report = analytics.generate_session_report(latest_session)
    print(report)

def show_help():
    """Show help and instructions"""
    help_text = """
    📖 SINGLE-ARM KETTLEBELL ROW TRACKER - HELP & INSTRUCTIONS
    ════════════════════════════════════════════════════════

    🎯 GETTING STARTED:
    1. Choose camera (live) or video file input
    2. Follow the on-screen instructions during your workout
    3. Focus on proper single-arm rowing form

    📹 CAMERA SETUP:
    • Position camera to capture your full body
    • Stand sideways to the camera for best tracking
    • Ensure good lighting and minimal background clutter
    • Test your camera before starting the workout

    🏋️ SINGLE-ARM ROW TECHNIQUE:
    • Pull elbow back past shoulder line
    • Squeeze shoulder blade at top of movement
    • Keep core engaged and torso stable
    • Control the weight on both up and down phases
    • Use your dominant arm for best tracking

    💻 CONTROLS DURING WORKOUT:
    • 'q' - Quit workout session
    • 'r' - Reset rep counter
    • 'p' - Pause/resume session
    • 's' - Take screenshot (camera mode)

    📊 VIEWING RESULTS:
    • Session logs are automatically saved
    • Use menu option 2 to see workout history
    • Use menu option 3 for detailed performance analysis

    🔧 TROUBLESHOOTING:
    • If pose detection fails, check lighting and positioning
    • Ensure your entire body is visible in the frame
    • Wear contrasting clothing for better detection
    • Close other applications using the camera
    • Make sure your working arm is clearly visible

    💡 TIPS FOR BEST RESULTS:
    • Start with lighter weights to focus on form
    • Pay attention to real-time feedback
    • Review session logs to track improvement
    • Maintain consistent workout schedule
    • Focus on slow, controlled movements

    ════════════════════════════════════════════════════════
    """
    print(help_text)

def main_menu():
    """Display main menu and handle user choices"""
    while True:
        print("\n" + "="*60)
        print("🏋️ SINGLE-ARM KETTLEBELL ROW TRACKER - MAIN MENU")
        print("="*60)
        print("1. 🚀 Start New Workout")
        print("2. 📊 View Past Sessions")
        print("3. 📈 Session Analytics")
        print("4. ❓ Help & Instructions")
        print("5. 🚪 Exit")
        print("="*60)
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            # Start new workout
            exercise_type = get_user_exercise_choice()
            input_type = get_user_input_choice()
            
            # Initialize tracker
            tracker = KettlebellWorkoutTracker(exercise_type)
            
            try:
                if input_type == 'camera':
                    tracker.run_webcam_session()
                else:  # video
                    video_path = get_video_file_path()
                    output_path = get_output_settings()
                    tracker.run_video_session(video_path, output_path, show_video=True)
            except Exception as e:
                print(f"❌ Error during workout session: {str(e)}")
                print("💡 Please check your input settings and try again.")
        
        elif choice == '2':
            view_session_history()
        
        elif choice == '3':
            view_session_analytics()
        
        elif choice == '4':
            show_help()
        
        elif choice == '5':
            print("\n👋 Thank you for using Single-Arm Kettlebell Row Tracker!")
            print("🏋️ Keep up the great work with your fitness journey!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")

# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================

def main():
    """Main application entry point"""
    print("🚀 Launching Single-Arm Kettlebell Row Tracker...")
    print("")
    
    # Display welcome screen
    display_welcome_screen()
    
    print("✅ All systems initialized successfully!")
    print("✅ Ready to track your single-arm kettlebell rows!")
    print("")
    
    # Start main menu
    main_menu()

if __name__ == "__main__":
    main()