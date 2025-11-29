import cv2
import sys
import dlib
import os
import numpy as np 
import math

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
    SCRIPT_DIR, '..', 'data', 'glasses.png' 
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

# NOTE: Keep the glasses image in its original orientation. If your
# `spectacles.png` currently appears upside-down, replace the image
# file with a corrected version or uncomment a rotation below to adjust.
# (Previously flipped here; removed to avoid inverted overlay.)
# Example alternatives you can try if needed:
# GLASSES_IMG = cv2.flip(GLASSES_IMG, 1)           # horizontal flip
# GLASSES_IMG = cv2.rotate(GLASSES_IMG, cv2.ROTATE_180)  # rotate 180 degrees
# ---------------------------------------------------------------------


# --- 2. Perspective Transformation Function ---
# Global smoothing state for transform stability across frames
SMOOTH_STATE = {
    'M': None,           # last affine matrix (2x3)
    'alpha': 0.45,       # smoothing factor (0-1), higher -> more smoothing
}


def overlay_glasses(frame, landmarks):
    """Compute a similarity/affine transform from glasses source anchors to
    eye/nose landmarks, apply temporal smoothing, and blend only inside the
    non-zero warped alpha bbox for better performance and fewer artifacts.
    """

    # Work on a local copy of the glasses image to avoid modifying the
    # global `GLASSES_IMG` and to prevent Python treating it as a local
    # variable when we rotate it below.
    glasses_img_local = GLASSES_IMG.copy()
    # 0. Separate the 4-channel image into BGR and Alpha (Mask)
    glasses_bgr = glasses_img_local[:, :, :3].copy()
    glasses_alpha = glasses_img_local[:, :, 3].copy()

    h, w = glasses_alpha.shape

    # 1. Define 3 source anchor points on the glasses image (left, right, center)
    #    These are empirical points that map to left-eye, right-eye, and nose.
    src_pts = np.float32([
        [w * 0.18, h * 0.60],   # left temple
        [w * 0.82, h * 0.60],   # right temple
        [w * 0.50, h * 0.75],   # nose / lower-center
    ])

    # 2. Compute destination points from dlib landmarks: use eye centers and nose
    pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

    left_eye_center = pts[36:42].mean(axis=0)  # landmarks 36-41
    right_eye_center = pts[42:48].mean(axis=0) # landmarks 42-47
    nose_center = pts[27]                      # landmark 27 is bridge of nose

    dst_pts = np.float32([
        left_eye_center,
        right_eye_center,
        nose_center
    ])

    # 3. Estimate a partial affine (similarity-like) transform from src->dst
    #    cv2.estimateAffinePartial2D maps src to dst (src,dst)
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    if M is None:
        # If failed, don't modify frame
        return frame

    # 3b. Auto-correct if the estimated rotation is close to upside-down
    #     Compute rotation angle from affine matrix. If it's > 90 degrees
    #     (i.e. image is likely upside-down), rotate the source image 180
    #     degrees and re-estimate the affine transform.
    try:
        angle_deg = math.degrees(math.atan2(M[0,1], M[0,0]))
    except Exception:
        angle_deg = 0.0

    if abs(angle_deg) > 90:
        # Rotate the local copy by 180 degrees to correct upside-down images
        glasses_img_local = cv2.rotate(glasses_img_local, cv2.ROTATE_180)

        # Recompute local copies and source anchors after rotation
        glasses_bgr = glasses_img_local[:, :, :3].copy()
        glasses_alpha = glasses_img_local[:, :, 3].copy()
        h, w = glasses_alpha.shape
        src_pts = np.float32([
            [w * 0.18, h * 0.60],   # left temple
            [w * 0.82, h * 0.60],   # right temple
            [w * 0.50, h * 0.75],   # nose / lower-center
        ])

        # Re-estimate the affine transform with the corrected source
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

        # If still None, bail out gracefully
        if M is None:
            return frame

    # 4. Smooth the affine matrix across frames for temporal stability
    alpha = SMOOTH_STATE.get('alpha', 0.45)
    if SMOOTH_STATE['M'] is None:
        SmoothedM = M
    else:
        SmoothedM = alpha * M + (1.0 - alpha) * SMOOTH_STATE['M']

    SMOOTH_STATE['M'] = SmoothedM

    # 5. Warp glasses BGR and alpha using the smoothed affine transform
    warped_bgr = cv2.warpAffine(glasses_bgr, SmoothedM, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped_alpha = cv2.warpAffine(glasses_alpha, SmoothedM, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 6. Create soft alpha mask (feather edges) and normalize
    #    Blur kernel size scales with face size for consistent feathering
    face_width = np.linalg.norm(right_eye_center - left_eye_center)
    k = int(max(3, min(31, face_width * 0.08)))
    if k % 2 == 0:
        k += 1

    alpha_mask = warped_alpha.astype(np.float32) / 255.0
    alpha_mask = cv2.GaussianBlur(alpha_mask, (k, k), 0)

    # 7. Find tight ROI of non-zero alpha to limit blending work
    ys, xs = np.where(alpha_mask > 0.01)
    if ys.size == 0 or xs.size == 0:
        return frame

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    # Clip ROI to frame
    y1 = max(0, y1); x1 = max(0, x1)
    y2 = min(frame.shape[0]-1, y2); x2 = min(frame.shape[1]-1, x2)

    roi = frame[y1:y2+1, x1:x2+1].astype(np.float32)
    fg = warped_bgr[y1:y2+1, x1:x2+1].astype(np.float32)
    a = alpha_mask[y1:y2+1, x1:x2+1][:, :, np.newaxis]

    # 8. Blend using alpha
    blended = fg * a + roi * (1.0 - a)
    frame[y1:y2+1, x1:x2+1] = blended.astype(np.uint8)

    return frame

# --- 3. Main Application Loop ---

def open_webcam():
    # Use camera index 1
    cap = cv2.VideoCapture(0) 

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