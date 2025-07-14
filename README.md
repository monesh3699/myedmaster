# 🏋️ Single-Arm Kettlebell Row Tracker

A comprehensive computer vision-based system for tracking single-arm kettlebell row exercises with real-time feedback, rep counting, and detailed workout analytics.

## 📋 Features

### 🎯 Real-time Exercise Tracking
- **Live Rep Counting**: Automatically counts repetitions using computer vision
- **Form Analysis**: Real-time feedback on exercise form and technique
- **Stage Detection**: Tracks "up" and "down" phases of the movement
- **Interactive Guidance**: On-screen instructions and motivational messages

### 📊 Comprehensive Analytics
- **Session Logging**: Detailed workout session data with timestamps
- **Performance Metrics**: Reps per minute, consistency scores, form quality
- **Progress Tracking**: Historical data and improvement analytics
- **Detailed Reports**: Comprehensive session analysis and feedback

### 💻 Multiple Input Sources
- **Live Camera**: Real-time tracking with webcam
- **Video Files**: Analyze pre-recorded workout videos
- **Output Recording**: Save analyzed videos with overlay data

### 🎨 Interactive Interface
- **Workout HUD**: Real-time display of reps, stage, and feedback
- **Progress Indicators**: Visual progress bars and session timers
- **Form Feedback Panel**: Live coaching tips and corrections
- **Pause/Resume**: Full session control capabilities

## 🚀 Getting Started

### Prerequisites

```bash
pip install opencv-python mediapipe numpy matplotlib
```

### Installation

1. Clone or download the `kettlebell_exercise_tracker.py` file
2. Ensure you have a camera connected (for live tracking)
3. Place any test videos in the same directory (optional)

### Running the Application

```bash
python kettlebell_exercise_tracker.py
```

## 🎯 Exercise Focus

This tracker is specifically designed for **Single-Arm Kettlebell Rows** and analyzes:

- **Elbow Pull-back**: Tracks elbow movement behind shoulder line
- **Shoulder Blade Engagement**: Monitors proper shoulder blade squeeze
- **Range of Motion**: Ensures full extension and contraction
- **Form Consistency**: Provides feedback on movement quality

## 📹 Setup Instructions

### Camera Setup
1. Position yourself sideways to the camera
2. Ensure your entire body is visible in the frame
3. Use good lighting and minimal background clutter
4. Keep your working arm clearly visible
5. Wear contrasting clothing for better detection

### Exercise Technique
- Pull elbow back past shoulder line
- Squeeze shoulder blade at top of movement
- Keep core engaged and torso stable
- Control the weight on both up and down phases
- Use your dominant arm for best tracking

## 🎮 Controls

### During Workout Session
- `q` - Quit workout session
- `r` - Reset rep counter
- `p` - Pause/resume session
- `s` - Take screenshot (camera mode)
- `SPACE` - Next frame (when paused in video mode)

## 📊 Session Analytics

### Automatic Logging
- All workout sessions are automatically saved
- Detailed frame-by-frame analysis
- Rep timestamps and stage transitions
- Form feedback and improvement suggestions

### Performance Metrics
- **Total Repetitions**: Complete rep count
- **Reps per Minute**: Workout intensity measure
- **Consistency Score**: Timing regularity (0-100)
- **Form Quality Score**: Based on real-time feedback (0-100)

### Data Storage
- Sessions saved in `workout_logs/` directory
- JSON format for easy analysis
- Separate files for session data and frame logs

## 🔧 Technical Details

### Computer Vision Pipeline
1. **Pose Detection**: Uses MediaPipe for real-time pose estimation
2. **Angle Calculation**: Computes joint angles using vector mathematics
3. **Stage Classification**: Determines exercise phases based on angle thresholds
4. **Rep Counting**: Tracks stage transitions for accurate counting

### Key Angles Monitored
- **Elbow Angle**: Shoulder-Elbow-Wrist angle
- **Shoulder Angle**: Hip-Shoulder-Elbow angle
- **Movement Range**: Configurable thresholds for each exercise phase

### Configuration
```python
SINGLE_ARM_ROW = {
    'elbow_angle_up': 160,         # Arm extended
    'elbow_angle_down': 80,        # Arm contracted
    'shoulder_angle_up': 20,       # Shoulder neutral
    'shoulder_angle_down': 45      # Shoulder pulled back
}
```

## 📈 Menu Options

### Main Menu
1. **🚀 Start New Workout** - Begin tracking session
2. **📊 View Past Sessions** - Review workout history
3. **📈 Session Analytics** - Detailed performance analysis
4. **❓ Help & Instructions** - Usage guide
5. **🚪 Exit** - Close application

### Input Sources
1. **Live Camera** - Real-time webcam tracking
2. **Video File** - Analyze pre-recorded videos

## 🔍 Troubleshooting

### Common Issues

**Camera Not Detected**
- Check camera permissions in system settings
- Close other applications using the camera
- Try different camera backends (automatically handled)

**Poor Pose Detection**
- Ensure good lighting conditions
- Position entire body in frame
- Wear contrasting clothing
- Check camera angle and distance

**Inaccurate Rep Counting**
- Verify proper exercise form
- Check camera positioning (sideways view)
- Ensure smooth, controlled movements
- Avoid rapid or jerky motions

## 📁 File Structure

```
kettlebell_exercise_tracker.py    # Main application file
workout_logs/                     # Session data directory
├── session_YYYYMMDD_HHMMSS.json # Session summaries
├── frames_YYYYMMDD_HHMMSS.json  # Detailed frame data
└── ...
```

## 🎯 Tips for Best Results

1. **Start with lighter weights** to focus on form
2. **Pay attention to real-time feedback** during workouts
3. **Review session logs** to track improvement
4. **Maintain consistent workout schedule**
5. **Focus on slow, controlled movements**
6. **Use your dominant arm** for best tracking accuracy

## 📊 Sample Output

```
🏁 WORKOUT SESSION COMPLETE
==================================================
📊 Total Reps: 15
⏱️ Duration: 3.2 minutes
🚀 Reps/Minute: 4.7
💬 Feedback Count: 23
💾 Session saved: workout_logs/session_20250714_143022.json
==================================================
```

## 🤝 Contributing

This is a specialized fitness tracking application. For improvements or bug reports, please ensure any modifications maintain the focus on defensive security practices and exercise form analysis.

## 📄 License

This project is provided for educational and personal fitness tracking purposes.

---

**Happy Training! 🏋️💪**

*Track your progress, perfect your form, and achieve your fitness goals with real-time computer vision feedback.*
