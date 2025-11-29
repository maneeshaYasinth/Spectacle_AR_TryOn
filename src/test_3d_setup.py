"""
Quick test script to verify 3D rendering setup
Run this before main_3d.py to check if everything is installed correctly
"""

import sys

def check_imports():
    """Check if all required packages are installed."""
    print("Checking dependencies...\n")
    
    packages = {
        'cv2': 'opencv-python',
        'dlib': 'dlib',
        'numpy': 'numpy',
        'trimesh': 'trimesh',
        'pyrender': 'pyrender',
        'PIL': 'Pillow'
    }
    
    missing = []
    
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"✓ {package:<20} OK")
        except ImportError:
            print(f"✗ {package:<20} MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True

def check_files():
    """Check if required data files exist."""
    import os
    
    print("\nChecking data files...\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    files = {
        'Facial landmarks': os.path.join(script_dir, '..', 'data', 'shape_predictor_68_face_landmarks.dat'),
        '3D model': os.path.join(script_dir, '..', 'data', 'glasses_model.obj')
    }
    
    missing = []
    
    for name, path in files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"✓ {name:<20} Found ({size:.1f} MB)")
        else:
            print(f"✗ {name:<20} MISSING")
            print(f"  Expected at: {path}")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        if '3D model' in missing:
            print("\n📦 Download a 3D glasses model from:")
            print("  - https://sketchfab.com (search 'glasses')")
            print("  - https://free3d.com")
            print("  Place it in: data/glasses_model.obj")
        if 'Facial landmarks' in missing:
            print("\n📦 Download dlib landmarks from:")
            print("  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("  Extract to: data/shape_predictor_68_face_landmarks.dat")
        return False
    else:
        print("\n✅ All required files found!")
        return True

def test_3d_rendering():
    """Test if pyrender can initialize."""
    print("\nTesting 3D rendering...\n")
    
    try:
        import trimesh
        import pyrender
        import numpy as np
        
        # Create a simple test mesh (cube)
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        py_mesh = pyrender.Mesh.from_trimesh(mesh)
        
        # Create scene
        scene = pyrender.Scene()
        scene.add(py_mesh)
        
        # Create camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=np.eye(4))
        
        # Try to create renderer
        renderer = pyrender.OffscreenRenderer(100, 100)
        
        # Test render
        color, depth = renderer.render(scene)
        
        renderer.delete()
        
        print("✓ 3D rendering test passed")
        print("✅ pyrender is working correctly!\n")
        return True
        
    except Exception as e:
        print(f"✗ 3D rendering test failed: {e}")
        print("\n❌ There may be an issue with OpenGL or pyrender setup")
        print("   Try: pip install --upgrade pyrender pyglet")
        return False

def main():
    """Run all checks."""
    print("=" * 60)
    print("  3D Glasses AR Try-On - Setup Verification")
    print("=" * 60 + "\n")
    
    checks = [
        check_imports(),
        check_files(),
        test_3d_rendering()
    ]
    
    print("=" * 60)
    if all(checks):
        print("🎉 SUCCESS! You're ready to run main_3d.py")
        print("\nRun the 3D version with:")
        print("  python src\\main_3d.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\nFor detailed setup instructions, see:")
        print("  3D_SETUP_GUIDE.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
