"""
3D Spectacle AR Try-On Application
Uses 3D model rendering instead of 2D image overlay
"""

import cv2
import sys
import dlib
import os
import numpy as np

# Import the 3D renderer
from renderer_3d import GlassesRenderer3D

# --- 1. Path Configuration and Model Loading ---

# Get the absolute directory where the current script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the absolute paths for the data files
PREDICTOR_DAT_PATH = os.path.join(
    SCRIPT_DIR, '..', 'data', 'shape_predictor_68_face_landmarks.dat'
)
GLASSES_MODEL_PATH = os.path.join(
    SCRIPT_DIR, '..', 'data', 'glasses_model.glb'  # or .obj, .stl, etc.
)

# Load the Dlib models with error handling
try:
    PREDICTOR = dlib.shape_predictor(PREDICTOR_DAT_PATH) 
    DLIB_DETECTOR = dlib.get_frontal_face_detector()
    print("Dlib models loaded successfully.")
except RuntimeError as e:
    print(f"Error loading Dlib model: {e}")
    print("FATAL: Please ensure 'shape_predictor_68_face_landmarks.dat' is in the 'data/' folder.")
    sys.exit()

# Check if 3D model exists
if not os.path.exists(GLASSES_MODEL_PATH):
    print(f"WARNING: 3D model not found at {GLASSES_MODEL_PATH}")
    print("Please place a 3D glasses model (.obj, .glb, .stl) in the data/ folder")
    print("You can download free models from:")
    print("  - https://sketchfab.com (search 'glasses' and filter by 'Downloadable')")
    print("  - https://free3d.com")
    print("  - https://www.thingiverse.com")
    sys.exit()

# Initialize 3D renderer (will be created after getting frame size)
RENDERER = None


# --- 2. Main Application Loop ---

def open_webcam():
    """Open webcam and run the AR try-on loop with 3D rendering."""
    global RENDERER
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

    print("Webcam opened successfully.")
    
    # Get frame dimensions
    ret, test_frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam.")
        sys.exit()
    
    frame_height, frame_width = test_frame.shape[:2]
    print(f"Frame size: {frame_width}x{frame_height}")
    
    # Initialize 3D renderer
    try:
        print("Initializing 3D renderer...")
        RENDERER = GlassesRenderer3D(
            GLASSES_MODEL_PATH,
            frame_width=frame_width,
            frame_height=frame_height
        )
        print("3D renderer initialized successfully!")
    except Exception as e:
        print(f"Error initializing 3D renderer: {e}")
        print("Make sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit()
    
    print("\n=== Controls ===")
    print("Press 'q' to quit")
    print("Press 's' to adjust smoothing (cycles through levels)")
    print("================\n")
    
    smoothing_levels = [0.3, 0.5, 0.7, 0.9]
    current_smoothing_idx = 1

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Mirror the frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using dlib
        faces = DLIB_DETECTOR(gray, 1)
        
        # Process each detected face
        for face in faces:
            # Get 68 facial landmarks
            landmarks = PREDICTOR(gray, face)
            
            # Render and overlay 3D glasses
            frame = RENDERER.overlay_on_frame(frame, landmarks)
            
            # Optional: Draw face rectangle for debugging
            # x, y, w, h = face.left(), face.top(), face.width(), face.height()
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display info on screen
        info_text = f"Smoothing: {RENDERER.smooth_alpha:.1f} | Faces: {len(faces)}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('3D Spectacle AR Try-On', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Cycle through smoothing levels
            current_smoothing_idx = (current_smoothing_idx + 1) % len(smoothing_levels)
            RENDERER.smooth_alpha = smoothing_levels[current_smoothing_idx]
            print(f"Smoothing set to: {RENDERER.smooth_alpha:.1f}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


# --- 3. Script Entry Point ---

if __name__ == "__main__":
    open_webcam()
