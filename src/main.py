import cv2
import sys
import dlib
import os
import numpy as np 

# --- 1. Path Configuration and Model Loading ---

# Get the absolute directory where the current script (main.py) is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the absolute paths for the data files
PREDICTOR_DAT_PATH = os.path.join(
    SCRIPT_DIR, '..', 'data', 'shape_predictor_68_face_landmarks.dat'
)
HAAR_CASCADE_PATH = os.path.join(
    SCRIPT_DIR, '..', 'data', 'cascades', 'haarcascade_frontalface_default.xml'
)
GLASSES_IMAGE_PATH = os.path.join(
    SCRIPT_DIR, '..', 'data', 'spectacles.png' 
)

# Load the Dlib models with error handling
try:
    PREDICTOR = dlib.shape_predictor(PREDICTOR_DAT_PATH) 
    DLIB_DETECTOR = dlib.get_frontal_face_detector()
    FACE_CASCADE = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
except RuntimeError as e:
    print(f"Error loading Dlib model: {e}")
    print("FATAL: Please ensure 'shape_predictor_68_face_landmarks.dat' is extracted and placed in the 'data/' folder.")
    sys.exit()

# Load the glasses image
GLASSES_IMG = cv2.imread(GLASSES_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

if GLASSES_IMG is None:
    print(f"FATAL ERROR: Could not load glasses image from {GLASSES_IMAGE_PATH}")
    sys.exit()


# --- CRITICAL FIX: Guaranteed 4-Channel Conversion ---
# Handles 2D/3D loading issues and guarantees 4 channels (BGRA) for blending.
if len(GLASSES_IMG.shape) == 2:
    print("FIX: Converting 2D Grayscale image to 4-channel BGRA for blending.")
    
    bgr_img = cv2.cvtColor(GLASSES_IMG, cv2.COLOR_GRAY2BGR)
    
    b, g, r = cv2.split(bgr_img)
    alpha = np.ones(b.shape, dtype=b.dtype) * 255
    GLASSES_IMG = cv2.merge((b, g, r, alpha))

elif GLASSES_IMG.shape[2] == 3:
    print("FIX: Converting 3-channel BGR image to 4-channel BGRA for blending.")
    b, g, r = cv2.split(GLASSES_IMG)
    alpha = np.ones(b.shape, dtype=b.dtype) * 255
    GLASSES_IMG = cv2.merge((b, g, r, alpha))

# >>> NEW FIX: Flip the image vertically to correct upside-down orientation <<<
GLASSES_IMG = cv2.flip(GLASSES_IMG, 0) 
# ---------------------------------------------------------------------


# --- 2. Perspective Transformation Function ---

def overlay_glasses(frame, landmarks):
    """Calculates perspective transform and overlays the glasses image onto the frame."""
    
    # 0. Separate the 4-channel image into BGR and Alpha (Mask)
    glasses_bgr = GLASSES_IMG[:, :, :3]  # The BGR color data (first 3 channels)
    glasses_mask = GLASSES_IMG[:, :, 3]   # The Alpha mask (the 4th channel)

    # Convert the 1-channel mask to 3-channel for warping purposes (standard practice)
    glasses_mask = cv2.cvtColor(glasses_mask, cv2.COLOR_GRAY2BGR)
    
    # 1. DEFINE SOURCE POINTS (Points on the original glasses PNG)
    h, w, _ = GLASSES_IMG.shape
    
    # *** ADJUSTED SOURCE POINTS FOR BETTER ALIGNMENT ***
    src_pts = np.float32([
        [int(w * 0.18), int(h * 0.6)],    # Left Temple (Side of glasses)
        [int(w * 0.82), int(h * 0.6)],    # Right Temple 
        [int(w * 0.35), int(h * 0.75)],   # Left Eye area 
        [int(w * 0.65), int(h * 0.75)]    # Right Eye area
    ])

    
    # 2. DEFINE DESTINATION POINTS (Points on the user's face, based on landmarks)
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 68)])
    
    dst_pts = np.float32([
        points[1],   # Left Temple 
        points[15],  # Right Temple
        points[36],  # Left Eye Inner Corner
        points[45]   # Right Eye Outer Corner
    ])
    
    # 3. CALCULATE THE PERSPECTIVE MATRIX (Homography)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 4. WARP THE BGR IMAGE AND THE MASK SEPARATELY (The Key Fix!)
    
    # Warp the BGR content of the glasses
    warped_bgr = cv2.warpPerspective(glasses_bgr, M, (frame.shape[1], frame.shape[0]))
    
    # Warp the 3-channel mask
    warped_mask_3c = cv2.warpPerspective(glasses_mask, M, (frame.shape[1], frame.shape[0]))
    
    # 5. BLEND THE WARPED IMAGE ONTO THE FRAME
    
    # Convert the 3-channel mask back to 1-channel float for blending math
    alpha_mask = cv2.cvtColor(warped_mask_3c, cv2.COLOR_BGR2GRAY) / 255.0
    
    # Since the mask might be blurry after warping, we sharpen it slightly
    alpha_mask[alpha_mask > 0.9] = 1.0
    alpha_mask[alpha_mask < 0.1] = 0.0
    
    alpha_mask_inv = 1.0 - alpha_mask
    
    # Ensure the alpha mask is a 3-channel array for element-wise multiplication
    alpha_mask_3c = cv2.merge((alpha_mask, alpha_mask, alpha_mask))
    alpha_mask_inv_3c = cv2.merge((alpha_mask_inv, alpha_mask_inv, alpha_mask_inv))

    # Apply the blending formula (Foreground * Mask + Background * Inverse Mask)
    # The warped BGR image is the foreground, and the current frame is the background.
    
    # Extract the region of interest (ROI) from the current frame
    h_w, w_w, _ = warped_bgr.shape
    roi = frame[0:h_w, 0:w_w]

    # Calculate the blended result
    foreground = cv2.multiply(warped_bgr.astype(float), alpha_mask_3c)
    background = cv2.multiply(roi.astype(float), alpha_mask_inv_3c)
    
    blended_result = cv2.add(foreground, background)
    
    # Copy the blended result back to the original frame
    frame[0:h_w, 0:w_w] = blended_result.astype(np.uint8)

    return frame

# --- 3. Main Application Loop ---

def open_webcam():
    # Use camera index 1
    cap = cv2.VideoCapture(1) 

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

    print("Webcam successfully opened. Press 'q' to exit.")

    while True:
        ret, frame = cap.read() 

        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Mirror the frame horizontally 
        frame = cv2.flip(frame, 1)

        # 1. Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Find faces using the dlib detector
        faces_dlib = DLIB_DETECTOR(gray, 1) 
        
        # 3. Process the detected faces
        for face in faces_dlib:
            # Predict the 68 facial landmarks
            landmarks = PREDICTOR(gray, face)
            
            # --- CORE AR STEP: Overlay the glasses ---
            frame = overlay_glasses(frame, landmarks)
            
                
        # Display the resulting frame
        cv2.imshow('Spectacle Try-On Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


# --- 4. Script Entry Point ---

if __name__ == "__main__":
    open_webcam()