import subprocess
import sys
import os
import webbrowser
import time
import threading

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def check_models():
    """Check if model files exist and provide fallbacks"""
    models_status = {
        "yolov8s_best.pt": False,
        "yolov11s_best.pt": False
    }
    
    model_paths = [
        get_resource_path('yolov8s_best.pt'),
        get_resource_path('yolov11s_best.pt'),
        'yolov8s_best.pt',
        'yolov11s_best.pt'
    ]
    
    print("🔍 Checking for model files...")
    
    for model_name in models_status.keys():
        for path in [get_resource_path(model_name), model_name]:
            if os.path.exists(path):
                models_status[model_name] = True
                print(f"   ✅ Found: {model_name}")
                break
        
        if not models_status[model_name]:
            print(f"   ❌ Missing: {model_name}")
    
    # Check if at least one model exists
    if not any(models_status.values()):
        print("⚠️  No trained models found. Downloading pretrained models...")
        try:
            from ultralytics import YOLO
            
            # Download pretrained models as fallbacks
            print("📥 Downloading YOLOv8s...")
            yolov8_model = YOLO('yolov8s.pt')
            yolov8_model.save('yolov8s_best.pt')
            
            print("📥 Downloading YOLOv11s...")
            yolov11_model = YOLO('yolo11s.pt')
            yolov11_model.save('yolov11s_best.pt')
            
            print("✅ Fallback models ready!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download fallback models: {e}")
            return False
    
    return True

def open_browser():
    """Open browser after delay"""
    time.sleep(4)  # Give Streamlit more time to start
    try:
        webbrowser.open('http://localhost:8501')
        print("🌐 Browser opened!")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("📱 Please open http://localhost:8501 manually")

def main():
    print("🪑 Furniture Detection App")
    print("=" * 50)
    print("🚀 Starting Dual Model YOLO App...")
    print("⚡ Powered by YOLOv8s & YOLOv11s")
    print("=" * 50)
    
    # Check for models
    if not check_models():
        print("\n❌ No models available. Cannot start app.")
        input("Press Enter to exit...")
        return
    
    # Get the path to the Streamlit app
    app_path = get_resource_path('furniture_detection_app.py')
    
    if not os.path.exists(app_path):
        print(f"❌ App file not found: {app_path}")
        input("Press Enter to exit...")
        return
    
    print(f"\n📱 Starting Streamlit app...")
    print(f"🔗 App will open at: http://localhost:8501")
    print(f"⏹️  To stop: Press Ctrl+C in this window")
    
    # Start browser in separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_path,
            '--server.headless', 'true',
            '--server.port', '8501',
            '--browser.gatherUsageStats', 'false',
            '--theme.backgroundColor', '#FFFFFF',
            '--theme.primaryColor', '#1f77b4'
        ])
    except KeyboardInterrupt:
        print("\n\n👋 App stopped by user")
        print("✅ Thank you for using Furniture Detection App!")
    except Exception as e:
        print(f"\n❌ Error starting app: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you have an internet connection")
        print("2. Check if port 8501 is available")
        print("3. Try running as administrator")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()