# 3D Glasses AR Try-On Setup Guide

## Overview
This project now supports **3D model rendering** for more realistic virtual try-on with proper perspective, rotation, and depth!

## What You Need

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Get a 3D Glasses Model

You need a 3D model file (`.obj`, `.glb`, `.stl`, or `.ply`) of glasses. Here's where to find free models:

#### Recommended Sources:
1. **Sketchfab** (https://sketchfab.com)
   - Search for "glasses" or "eyeglasses"
   - Filter by "Downloadable"
   - Look for CC-licensed models
   - Download in `.glb` or `.obj` format

2. **Free3D** (https://free3d.com)
   - Search "glasses"
   - Download `.obj` or `.fbx` format

3. **Thingiverse** (https://www.thingiverse.com)
   - Search "glasses" or "spectacles"
   - Download `.stl` files

4. **CGTrader** (https://www.cgtrader.com/free-3d-models)
   - Filter for free models
   - Search "eyeglasses"

#### Quick Example:
- Download this model: [Simple Glasses on Sketchfab](https://sketchfab.com/3d-models/glasses-d91ac66e5e444f7d92e38d0e5c32cbb8)
- Click "Download 3D Model" → Select format (GLB or OBJ)
- Extract and save to: `data/glasses_model.obj` (or `.glb`)

### 3. Place the Model File
Put your downloaded 3D model file in the `data/` folder:
```
Spectacle_AR_TryOn/
├── data/
│   ├── glasses_model.obj      ← Place your model here
│   ├── shape_predictor_68_face_landmarks.dat
│   └── cascades/
├── src/
│   ├── main_3d.py             ← New 3D rendering version
│   ├── renderer_3d.py         ← 3D renderer module
│   └── main.py                ← Original 2D version (still works)
```

**Important:** The default filename is `glasses_model.obj`. If your file has a different name or extension, either:
- Rename it to `glasses_model.obj`, OR
- Edit `src/main_3d.py` line 21 to match your filename

## Running the 3D Version

### Option A: Run the new 3D version
```powershell
python src\main_3d.py
```

### Option B: Keep using the 2D version
```powershell
python src\main.py
```

## Controls (3D Version)
- **Q**: Quit the application
- **S**: Cycle through smoothing levels (0.3 → 0.5 → 0.7 → 0.9)

## How It Works

### 2D Version (`main.py`)
- Loads a PNG image with transparency
- Uses affine transformation to fit to face
- Fast but limited perspective

### 3D Version (`main_3d.py`)
- Loads a full 3D model with geometry
- Estimates head pose in 3D space using `cv2.solvePnP`
- Renders the model with proper perspective using `pyrender`
- Realistic rotation, scale, and depth
- Smooth temporal filtering to reduce jitter

## Troubleshooting

### "Module not found" errors
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### "Could not load 3D model"
- Ensure the model file is in `data/` folder
- Check the filename matches (default: `glasses_model.obj`)
- Try different model formats (`.glb` is most reliable)

### Model appears too large/small
Edit `src/renderer_3d.py` line 66-68 to adjust scale:
```python
scale_factor = 140.0 / current_width  # Change 140.0 to larger/smaller value
```

### Model is upside down or rotated wrong
Add rotation to `src/renderer_3d.py` after line 68:
```python
# Rotate 180° around X axis
rotation = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
self.mesh.apply_transform(rotation)
```

### Glasses don't align with face
The 6-point correspondence in `renderer_3d.py` uses standard facial landmarks. You may need to:
1. Adjust `model_points_3d` (lines 45-52) for better alignment
2. Tweak smoothing with the 'S' key during runtime
3. Use a different model with better proportions

### Performance issues
- Use simpler models (fewer polygons)
- Reduce frame size in renderer initialization
- Close other applications using GPU

## Model Preparation Tips

### If your model needs adjustment:
Use Blender (free) to:
1. Center the model (Object → Set Origin → Geometry to Origin)
2. Scale appropriately (typical glasses: ~140mm width)
3. Rotate to face -Z direction
4. Export as `.obj` or `.glb`

### Optimal model characteristics:
- **Polygon count**: 1,000-10,000 triangles (lower = faster)
- **Format**: `.glb` (single file) or `.obj` with `.mtl`
- **Textures**: Keep texture files in same folder as model
- **Scale**: Real-world scale (mm) works best
- **Orientation**: Front of glasses facing -Z axis

## Next Steps

### Add multiple glasses models:
1. Create multiple model files: `glasses1.obj`, `glasses2.obj`, etc.
2. Edit `main_3d.py` to switch between models
3. Add keyboard controls to cycle through options

### Improve lighting:
Edit `renderer_3d.py` lines 127-131 to add more lights:
```python
# Add multiple lights for better illumination
light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
light2 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=5.0)
self.scene.add(light1, pose=np.eye(4))
self.scene.add(light2, pose=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,500],[0,0,0,1]]))
```

### Add reflections/materials:
Use models with PBR textures (metallic/roughness maps) for realistic materials.

## Comparison: 2D vs 3D

| Feature | 2D (main.py) | 3D (main_3d.py) |
|---------|--------------|-----------------|
| Setup | Easy (PNG only) | Moderate (need 3D model) |
| Performance | Very fast | Fast (GPU dependent) |
| Realism | Good for flat view | Excellent with rotation |
| Perspective | Limited | Full 3D perspective |
| Customization | Change PNG | Full 3D editing possible |
| File size | Small (~KB) | Larger (~MB) |

## Dependencies Explained

- **trimesh**: Load and manipulate 3D models
- **pyrender**: OpenGL-based offscreen rendering
- **opencv-python**: Camera, face detection, pose estimation
- **dlib**: 68-point facial landmark detection
- **numpy**: Numerical operations
- **Pillow**: Image processing support
- **pyglet<2**: Required by pyrender (v2 has compatibility issues)

## Support

If you encounter issues:
1. Check that all dependencies installed correctly
2. Verify your 3D model loads in Blender or another viewer
3. Try the 2D version first to ensure face detection works
4. Check Python version (tested on Python 3.8-3.11)

Enjoy your realistic 3D AR glasses try-on! 🕶️
