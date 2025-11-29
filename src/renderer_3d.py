"""
3D Glasses Renderer using Trimesh + Pyrender
Handles head pose estimation from facial landmarks and renders 3D models
"""

import numpy as np
import cv2
import trimesh
import pyrender
import os


class GlassesRenderer3D:
    """Renders 3D glasses model onto video frames using head pose estimation."""
    
    def __init__(self, model_path, frame_width=640, frame_height=480):
        """
        Initialize the 3D renderer.
        
        Args:
            model_path: Path to 3D model file (.obj, .glb, .stl, etc.)
            frame_width: Width of video frame for camera matrix
            frame_height: Height of video frame for camera matrix
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Load 3D model
        print(f"Loading 3D model from: {model_path}")
        try:
            self.mesh = trimesh.load(model_path)
            
            # If it's a Scene (multiple meshes), merge them
            if isinstance(self.mesh, trimesh.Scene):
                self.mesh = trimesh.util.concatenate(
                    [geom for geom in self.mesh.geometry.values()]
                )
            
            # Center and scale the model for better fit
            self._prepare_model()
            
            print(f"Model loaded successfully: {self.mesh.vertices.shape[0]} vertices")
        except Exception as e:
            raise RuntimeError(f"Failed to load 3D model: {e}")
        
        # Define 3D model points for key facial landmarks (in model space)
        # These correspond to: nose tip, chin, left eye outer, right eye outer,
        # left mouth corner, right mouth corner
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye outer corner
            (225.0, 170.0, -135.0),   # Right eye outer corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ], dtype=np.float64)
        
        # Camera matrix (estimated for typical webcam)
        self.camera_matrix = self._get_camera_matrix()
        
        # Distortion coefficients (assuming no distortion for simplicity)
        self.dist_coeffs = np.zeros((4, 1))
        
        # Smoothing state for pose
        self.smooth_rotation = None
        self.smooth_translation = None
        self.smooth_alpha = 0.5  # Smoothing factor (0=no smoothing, 1=max smoothing)
        
        # Initialize pyrender scene
        self._init_renderer()
    
    def _prepare_model(self):
        """Center and scale the 3D model for proper positioning."""
        # Center the model
        center = self.mesh.bounds.mean(axis=0)
        self.mesh.vertices -= center
        
        # Scale to reasonable size (adjust as needed)
        # Typical glasses are about 140mm wide
        current_width = self.mesh.bounds[1][0] - self.mesh.bounds[0][0]
        if current_width > 0:
            scale_factor = 140.0 / current_width
            self.mesh.vertices *= scale_factor
        
        # Rotate to align with face coordinate system if needed
        # (Assuming model is facing -Z direction initially)
        # Adjust this based on your specific model orientation
    
    def _get_camera_matrix(self):
        """Create camera intrinsic matrix."""
        focal_length = self.frame_width
        center = (self.frame_width / 2, self.frame_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        return camera_matrix
    
    def _init_renderer(self):
        """Initialize pyrender offscreen renderer."""
        # Create pyrender mesh from trimesh
        self.py_mesh = pyrender.Mesh.from_trimesh(self.mesh, smooth=True)
        
        # Create scene
        self.scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0, 0, 0, 0])
        
        # Create camera (will be updated each frame)
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        self.camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy,
            znear=0.05, zfar=10000.0
        )
        
        # Add directional light for better visualization
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        
        self.scene.add(self.camera, name='camera')
        self.scene.add(light, pose=np.eye(4))
        
        # Offscreen renderer
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.frame_width,
            viewport_height=self.frame_height
        )
        
        # Node references
        self.mesh_node = None
        self.camera_node = self.scene.get_nodes(name='camera')[0]
    
    def estimate_head_pose(self, landmarks):
        """
        Estimate head pose from facial landmarks using solvePnP.
        
        Args:
            landmarks: dlib shape predictor output (68 landmarks)
            
        Returns:
            (rotation_vector, translation_vector) or (None, None) if failed
        """
        # Extract 2D image points from landmarks
        # Indices: 30 (nose tip), 8 (chin), 36 (left eye outer),
        #          45 (right eye outer), 48 (left mouth), 54 (right mouth)
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye outer
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye outer
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
        ], dtype=np.float64)
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_3d,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, None
        
        # Apply temporal smoothing
        if self.smooth_rotation is None:
            self.smooth_rotation = rotation_vector
            self.smooth_translation = translation_vector
        else:
            alpha = self.smooth_alpha
            self.smooth_rotation = alpha * rotation_vector + (1 - alpha) * self.smooth_rotation
            self.smooth_translation = alpha * translation_vector + (1 - alpha) * self.smooth_translation
        
        return self.smooth_rotation.copy(), self.smooth_translation.copy()
    
    def render(self, rotation_vector, translation_vector):
        """
        Render the 3D glasses from the estimated pose.
        
        Args:
            rotation_vector: 3x1 rotation vector from solvePnP
            translation_vector: 3x1 translation vector from solvePnP
            
        Returns:
            (color, alpha) tuple - RGBA rendered image split into RGB and A
        """
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Create 4x4 transformation matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = translation_vector.flatten()
        
        # Remove old mesh node if exists
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        
        # Add mesh with new pose
        self.mesh_node = self.scene.add(self.py_mesh, pose=pose_matrix)
        
        # Render
        color, depth = self.renderer.render(self.scene)
        
        # Split into RGB and alpha
        # Create alpha channel from depth (non-zero depth = opaque)
        alpha = (depth > 0).astype(np.uint8) * 255
        
        return color, alpha
    
    def overlay_on_frame(self, frame, landmarks):
        """
        Complete pipeline: estimate pose, render, and composite onto frame.
        
        Args:
            frame: Input video frame (BGR)
            landmarks: dlib facial landmarks
            
        Returns:
            frame with 3D glasses overlaid
        """
        # Estimate head pose
        rvec, tvec = self.estimate_head_pose(landmarks)
        
        if rvec is None or tvec is None:
            return frame
        
        # Render 3D model
        try:
            color, alpha = self.render(rvec, tvec)
        except Exception as e:
            print(f"Rendering error: {e}")
            return frame
        
        # Convert color from RGB to BGR for OpenCV
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        
        # Blend onto frame using alpha
        alpha_mask = alpha.astype(np.float32) / 255.0
        
        # Apply Gaussian blur for softer edges
        alpha_mask = cv2.GaussianBlur(alpha_mask, (5, 5), 0)
        
        # Expand alpha to 3 channels
        alpha_3ch = np.stack([alpha_mask] * 3, axis=-1)
        
        # Blend
        frame_float = frame.astype(np.float32)
        color_float = color_bgr.astype(np.float32)
        
        blended = color_float * alpha_3ch + frame_float * (1 - alpha_3ch)
        
        return blended.astype(np.uint8)
    
    def __del__(self):
        """Cleanup renderer resources."""
        if hasattr(self, 'renderer'):
            self.renderer.delete()
