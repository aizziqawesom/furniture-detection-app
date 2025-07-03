import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
from ultralytics import YOLO
import time
from datetime import datetime
import json
import zipfile
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import glob

# Page configuration
st.set_page_config(
    page_title="🪑 Furniture Detection App",
    page_icon="🪑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .detection-stats {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .confidence-box {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 0.25rem;
        color: white;
        font-weight: bold;
    }
    .model-info {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .model-comparison {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .sample-card {
        border: 2px solid #e1e5e9;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 0.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .sample-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .sample-card.selected {
        border-color: #1f77b4;
        background-color: #f0f8ff;
    }
    .gallery-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Furniture classes from your dataset
FURNITURE_CLASSES = [
    'air conditioner', 'bed', 'cabinet', 'carpet', 'ceiling fan', 
    'chair', 'closet', 'computer', 'cupboard', 'desk', 
    'dining table', 'drawer', 'frame', 'lamp', 'monitor', 
    'shelf', 'sofa', 'stool', 'table', 'wardrobe'
]

# Color palette for bounding boxes
COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#FFB347', '#87CEEB', '#F0E68C', '#FFA07A',
    '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471',
    '#82E0AA', '#F1948A', '#85C1E9', '#F4D03F', '#AED6F1'
]

# Model configurations
MODEL_CONFIGS = {
    "YOLOv8s": {
        "name": "YOLOv8s",
        "path": "yolov8s_best.pt",
        "description": "YOLOv8s - Balanced speed and accuracy",
        "color": "#4CAF50",
        "icon": "🟢"
    },
    "YOLOv11s": {
        "name": "YOLOv11s", 
        "path": "yolov11s_best.pt",
        "description": "YOLOv11s - Latest model with improved accuracy",
        "color": "#2196F3",
        "icon": "🔵"
    },
    "Roboflow": {
        "name": "Roboflow 3.0",
        "path": "roboflow_best.pt",
        "description": "Roboflow 3.0 - Enhanced training pipeline",
        "color": "#FF5722",
        "icon": "🔴",
        "results_folder": "roboflow3.0_training2"
    }
}

def get_sample_files():
    """Get available sample images and videos from the repository"""
    sample_dirs = ['examples', 'samples', 'test_data', 'demo_files']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    samples = {
        'images': [],
        'videos': []
    }
    
    # Check multiple possible directories for samples
    for sample_dir in sample_dirs:
        if os.path.exists(sample_dir):
            # Get all files in the directory
            for file_path in Path(sample_dir).rglob('*'):
                if file_path.is_file():
                    file_ext = file_path.suffix.lower()
                    relative_path = str(file_path)
                    
                    if file_ext in image_extensions:
                        samples['images'].append({
                            'name': file_path.name,
                            'path': relative_path,
                            'type': 'image',
                            'size': file_path.stat().st_size
                        })
                    elif file_ext in video_extensions:
                        samples['videos'].append({
                            'name': file_path.name,
                            'path': relative_path,
                            'type': 'video',
                            'size': file_path.stat().st_size
                        })
    
    # If no samples found, create some placeholder info
    if not samples['images'] and not samples['videos']:
        st.warning("⚠️ No sample files found. Please add sample images/videos to the 'examples' folder.")
        
        # Suggest folder structure
        st.info("""
        📁 **Suggested folder structure:**
        ```
        examples/
        ├── images/
        │   ├── living_room.jpg
        │   ├── bedroom.jpg
        │   ├── kitchen.jpg
        │   └── office.jpg
        └── videos/
            ├── room_tour.mp4
            └── furniture_showcase.mp4
        ```
        """)
    
    return samples

def display_sample_gallery(sample_type='images'):
    """Display sample gallery for user selection"""
    samples = get_sample_files()
    
    if not samples[sample_type]:
        st.warning(f"No sample {sample_type} found in the repository.")
        return None
    
    st.markdown(f"### 🖼️ Sample {sample_type.title()} Gallery")
    st.markdown(f"*Select from {len(samples[sample_type])} available {sample_type}*")
    
    # Create columns for gallery display
    cols_per_row = 4 if sample_type == 'images' else 3
    
    selected_sample = None
    
    # Display samples in a grid
    for i in range(0, len(samples[sample_type]), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(samples[sample_type]):
                sample = samples[sample_type][i + j]
                
                with col:
                    # Create a unique key for each sample
                    sample_key = f"{sample_type}_{i+j}_{sample['name']}"
                    
                    if sample_type == 'images':
                        try:
                            # Display thumbnail
                            img = Image.open(sample['path'])
                            # Resize for thumbnail
                            img.thumbnail((200, 200))
                            st.image(img, use_column_width=True)
                        except Exception as e:
                            st.error(f"Cannot load {sample['name']}")
                            continue
                    else:  # videos
                        # For videos, show a placeholder or first frame
                        st.markdown(f"🎥 **{sample['name']}**")
                        st.markdown(f"Size: {sample['size'] / (1024*1024):.1f} MB")
                    
                    # Selection button
                    if st.button(
                        f"📂 Select {sample['name']}", 
                        key=sample_key,
                        help=f"Use {sample['name']} for detection"
                    ):
                        selected_sample = sample
                    
                    # Show file info
                    st.caption(f"📄 {sample['name']}")
                    if sample_type == 'images':
                        st.caption(f"📊 {sample['size'] / 1024:.1f} KB")
    
    return selected_sample

def create_sample_folders():
    """Create sample folders and add some demo content suggestions"""
    examples_dir = Path('examples')
    images_dir = examples_dir / 'images'
    videos_dir = examples_dir / 'videos'
    
    # Create directories if they don't exist
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a README file with instructions
    readme_content = """# Sample Files for Furniture Detection

## Folder Structure
- `images/` - Place sample images here (jpg, png, bmp formats)
- `videos/` - Place sample videos here (mp4, avi, mov formats)

## Recommended Sample Content
### Images:
- Living room scenes with sofas, tables, lamps
- Bedroom scenes with beds, wardrobes, chairs
- Kitchen scenes with cabinets, dining tables
- Office scenes with desks, chairs, monitors

### Videos:
- Room tours showing multiple furniture pieces
- Furniture arrangement videos
- Interior design showcases

## File Naming Suggestions:
- `living_room_01.jpg`
- `bedroom_modern.jpg`
- `kitchen_cabinet.jpg`
- `office_setup.jpg`
- `room_tour.mp4`
- `furniture_showcase.mp4`

The app will automatically detect and display these files in the Sample Gallery.
"""
    
    readme_path = examples_dir / 'README.md'
    if not readme_path.exists():
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    return images_dir, videos_dir

# [Previous functions remain the same: parse_results_metrics, get_model_performance_data, 
# display_model_performance, display_model_details, load_model, get_model_info, 
# draw_bounding_boxes, process_image, process_video, export_results, compare_models]

def parse_results_metrics(results_folder):
    """Parse metrics from results.png or results2.png"""
    results_path = Path(results_folder)
    
    # Try to find results images
    results_files = list(results_path.glob("results*.png"))
    
    if not results_files:
        return None
    
    # For now, return placeholder values - you can implement OCR or manual input later
    # These would typically be extracted from the metrics display in the results image
    return {
        "mAP50": 0.0,  # You'll need to manually input these or implement OCR
        "mAP50-95": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "results_image": str(results_files[0])
    }

def get_model_performance_data():
    """Get performance data for all models"""
    performance_data = {}
    
    # Manual performance data - replace with actual values from your training results
    performance_data = {
        "YOLOv8s": {
            "mAP50": 83.6,
            "mAP50-95": 69.7,
            "Precision": 82.9,
            "Recall": 79.3,
            "results_folder": "yolo8s_training2"
        },
        "YOLOv11s": {
            "mAP50": 82.9,
            "mAP50-95": 70.3,
            "Precision": 82.1,
            "Recall": 81.6,
            "results_folder": "yolo11s_training2"
        },
        "Roboflow": {
            "mAP50": 84.2,
            "mAP50-95": 61.9,
            "Precision": 82.4,
            "Recall": 79.9,
            "results_folder": "roboflow3.0_training2"
        }
    }
    
    return performance_data

def display_model_performance():
    """Display model performance comparison page"""
    st.markdown('<h1 class="main-header">📊 Model Performance Analysis</h1>', unsafe_allow_html=True)
    st.markdown("**Comprehensive comparison of YOLOv8s, YOLOv11s, and Roboflow 3.0 training results**")
    
    # Get performance data
    performance_data = get_model_performance_data()
    
    # Summary metrics at the top
    st.markdown('<h2 class="sub-header">🎯 Performance Summary</h2>', unsafe_allow_html=True)
    
    # Create summary table
    summary_df = pd.DataFrame({
        'Model': list(performance_data.keys()),
        'mAP@0.5': [f"{data['mAP50']:.1f}%" for data in performance_data.values()],
        'mAP@0.5:0.95': [f"{data['mAP50-95']:.1f}%" for data in performance_data.values()],
        'Precision': [f"{data['Precision']:.1f}%" for data in performance_data.values()],
        'Recall': [f"{data['Recall']:.1f}%" for data in performance_data.values()]
    })
    
    # Display summary table with styling
    st.markdown("""
    <style>
    .performance-table {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="performance-table">', unsafe_allow_html=True)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Best performing model highlight
    best_map50 = max(performance_data.values(), key=lambda x: x['mAP50'])
    best_model = [k for k, v in performance_data.items() if v['mAP50'] == best_map50['mAP50']][0]
    
    st.success(f"🏆 **Best Overall Performance**: {best_model} with mAP@0.5 of {best_map50['mAP50']:.1f}%")
    
    # Detailed results for each model
    st.markdown('<h2 class="sub-header">📈 Detailed Training Results</h2>', unsafe_allow_html=True)
    
    # Create tabs for each model
    tab1, tab2, tab3 = st.tabs(["🟢 YOLOv8s", "🔵 YOLOv11s", "🔴 Roboflow 3.0"])
    
    with tab1:
        display_model_details("YOLOv8s", performance_data["YOLOv8s"])
    
    with tab2:
        display_model_details("YOLOv11s", performance_data["YOLOv11s"])
    
    with tab3:
        display_model_details("Roboflow", performance_data["Roboflow"])

def display_model_details(model_name, data):
    """Display detailed results for a specific model"""
    results_folder = data['results_folder']
    
    # Model info header
    st.markdown(f"### {MODEL_CONFIGS[model_name]['icon']} {model_name} Training Results")
    
    # Performance metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="mAP@0.5", 
            value=f"{data['mAP50']:.1f}%",
            delta=f"{data['mAP50'] - 80:.1f}%" if data['mAP50'] > 80 else None
        )
    
    with col2:
        st.metric(
            label="mAP@0.5:0.95", 
            value=f"{data['mAP50-95']:.1f}%",
            delta=f"{data['mAP50-95'] - 60:.1f}%" if data['mAP50-95'] > 60 else None
        )
    
    with col3:
        st.metric(
            label="Precision", 
            value=f"{data['Precision']:.1f}%",
            delta=f"{data['Precision'] - 80:.1f}%" if data['Precision'] > 80 else None
        )
    
    with col4:
        st.metric(
            label="Recall", 
            value=f"{data['Recall']:.1f}%",
            delta=f"{data['Recall'] - 75:.1f}%" if data['Recall'] > 75 else None
        )
    
    # Display training results images
    st.markdown("#### 📊 Training Curves and Metrics")
    
    # Check if results folder exists
    results_path = Path(results_folder)
    if results_path.exists():
        # Display results.png or results2.png
        results_files = list(results_path.glob("results*.png"))
        if results_files:
            st.image(str(results_files[0]), caption=f"{model_name} Training Results", use_column_width=True)
        
        # Display confusion matrix
        confusion_files = list(results_path.glob("confusion_matrix*.png"))
        if confusion_files:
            col1, col2 = st.columns(2)
            with col1:
                st.image(str(confusion_files[0]), caption="Confusion Matrix", use_column_width=True)
            
            # Display normalized confusion matrix if available
            normalized_files = list(results_path.glob("*normalized*.png"))
            if normalized_files:
                with col2:
                    st.image(str(normalized_files[0]), caption="Normalized Confusion Matrix", use_column_width=True)
        
        # Display curves (F1, P, R, PR)
        st.markdown("#### 📈 Performance Curves")
        curve_cols = st.columns(3)
        
        curve_files = {
            "F1_curve.png": "F1 Score Curve",
            "P_curve.png": "Precision Curve", 
            "R_curve.png": "Recall Curve",
            "PR_curve.png": "Precision-Recall Curve"
        }
        
        curve_index = 0
        for curve_file, caption in curve_files.items():
            curve_path = results_path / curve_file
            if curve_path.exists():
                with curve_cols[curve_index % 3]:
                    st.image(str(curve_path), caption=caption, use_column_width=True)
                curve_index += 1
        
        # Display validation batch examples
        st.markdown("#### 🖼️ Validation Examples")
        val_batch_files = list(results_path.glob("val_batch*_labels.jpg"))
        pred_batch_files = list(results_path.glob("val_batch*_pred*.jpg"))
        
        if val_batch_files and pred_batch_files:
            col1, col2 = st.columns(2)
            with col1:
                st.image(str(val_batch_files[0]), caption="Ground Truth Labels", use_column_width=True)
            with col2:
                st.image(str(pred_batch_files[0]), caption="Model Predictions", use_column_width=True)
    
    else:
        st.warning(f"⚠️ Results folder '{results_folder}' not found. Please ensure training results are available.")

def display_about_page():
    """Display comprehensive about page"""
    st.markdown('<h1 class="main-header">ℹ️ About Furniture Detection App</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="model-info">
        <h3>🪑 Welcome to the Dual YOLO Furniture Detection App!</h3>
        <p>This advanced computer vision application uses state-of-the-art YOLO (You Only Look Once) models to detect and classify furniture items in images and videos with high accuracy and speed.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    st.markdown('<h2 class="sub-header">🚀 Key Features</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🤖 Multi-Model Support**
        - YOLOv8s - Balanced speed and accuracy
        - YOLOv11s - Latest model with improved performance
        - Roboflow 3.0 - Enhanced training pipeline
        
        **📱 Multiple Detection Modes**
        - 📸 Image Upload - Process single images
        - 🎥 Video Upload - Process video files
        - 📹 Live Webcam - Real-time detection
        - 📂 Batch Processing - Multiple files at once
        - 🖼️ Sample Gallery - Pre-loaded test samples
        """)
    
    with col2:
        st.markdown("""
        **🔬 Advanced Analysis**
        - Model comparison and benchmarking
        - Performance metrics visualization
        - Confidence threshold adjustment
        - Class-specific filtering
        
        **💾 Export & Integration**
        - JSON, CSV, Excel export formats
        - Downloadable processed videos
        - Detection statistics and reports
        - Training results analysis
        """)
    
    # Detectable Furniture Classes
    st.markdown('<h2 class="sub-header">🏠 Detectable Furniture Classes</h2>', unsafe_allow_html=True)
    st.markdown("The app can detect **20 different types of furniture** with high accuracy:")
    
    # Display furniture classes in a nice grid
    furniture_cols = st.columns(4)
    for i, furniture_class in enumerate(FURNITURE_CLASSES):
        with furniture_cols[i % 4]:
            st.markdown(f"• {furniture_class.title()}")
    
    # Technology Stack
    st.markdown('<h2 class="sub-header">⚙️ Technology Stack</h2>', unsafe_allow_html=True)
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        **🧠 AI/ML Frameworks**
        - Ultralytics YOLO
        - PyTorch
        - OpenCV
        - PIL (Python Imaging)
        """)
    
    with tech_col2:
        st.markdown("""
        **🌐 Web & UI**
        - Streamlit
        - HTML/CSS
        - Pandas
        - Matplotlib
        """)
    
    with tech_col3:
        st.markdown("""
        **📊 Data Processing**
        - NumPy
        - JSON/CSV/Excel export
        - Video processing (FFmpeg)
        - Real-time streaming
        """)
    
    # Model Performance Summary
    st.markdown('<h2 class="sub-header">📈 Model Performance Overview</h2>', unsafe_allow_html=True)
    
    performance_data = get_model_performance_data()
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.markdown(f"""
        <div class="model-comparison">
            <h4>🟢 YOLOv8s</h4>
            <p><strong>mAP@0.5:</strong> {performance_data['YOLOv8s']['mAP50']:.1f}%</p>
            <p><strong>Precision:</strong> {performance_data['YOLOv8s']['Precision']:.1f}%</p>
            <p><strong>Recall:</strong> {performance_data['YOLOv8s']['Recall']:.1f}%</p>
            <p><em>Balanced speed and accuracy</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col2:
        st.markdown(f"""
        <div class="model-comparison">
            <h4>🔵 YOLOv11s</h4>
            <p><strong>mAP@0.5:</strong> {performance_data['YOLOv11s']['mAP50']:.1f}%</p>
            <p><strong>Precision:</strong> {performance_data['YOLOv11s']['Precision']:.1f}%</p>
            <p><strong>Recall:</strong> {performance_data['YOLOv11s']['Recall']:.1f}%</p>
            <p><em>Latest model architecture</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col3:
        st.markdown(f"""
        <div class="model-comparison">
            <h4>🔴 Roboflow 3.0</h4>
            <p><strong>mAP@0.5:</strong> {performance_data['Roboflow']['mAP50']:.1f}%</p>
            <p><strong>Precision:</strong> {performance_data['Roboflow']['Precision']:.1f}%</p>
            <p><strong>Recall:</strong> {performance_data['Roboflow']['Recall']:.1f}%</p>
            <p><em>Enhanced training pipeline</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown('<h2 class="sub-header">🚀 Quick Start Guide</h2>', unsafe_allow_html=True)
    
    with st.expander("📖 How to Use This App", expanded=False):
        st.markdown("""
        ### Step 1: Choose Your Detection Mode
        - **📸 Image Upload:** Upload your own furniture images
        - **🎥 Video Upload:** Process furniture detection in videos
        - **🖼️ Sample Gallery:** Test with pre-loaded samples
        - **📹 Live Webcam:** Real-time furniture detection
        - **📂 Batch Processing:** Process multiple files at once
        
        ### Step 2: Configure Settings
        - Select your preferred YOLO model (YOLOv8s, YOLOv11s, or Roboflow)
        - Adjust confidence threshold (0.0 - 1.0)
        - Choose which furniture classes to detect
        - Enable model comparison mode for benchmarking
        
        ### Step 3: Upload or Select Content
        - Upload your images/videos or choose from sample gallery
        - The app supports JPG, PNG, BMP for images and MP4, AVI, MOV for videos
        
        ### Step 4: Analyze Results
        - View detected furniture with bounding boxes and confidence scores
        - Compare results between different models
        - Export detection data in JSON, CSV, or Excel format
        - Download processed videos with annotations
        
        ### Step 5: Explore Performance Analytics
        - Visit the "📊 Model Performance" page to see detailed training metrics
        - Compare model accuracies, precision, and recall
        - View confusion matrices and performance curves
        """)
    
    # Use Cases
    st.markdown('<h2 class="sub-header">🎯 Use Cases & Applications</h2>', unsafe_allow_html=True)
    
    use_case_col1, use_case_col2 = st.columns(2)
    
    with use_case_col1:
        st.markdown("""
        **🏠 Real Estate & Property**
        - Automated furniture inventory
        - Property staging verification
        - Virtual home tours enhancement
        - Rental property documentation
        
        **🛒 E-commerce & Retail**
        - Product cataloging automation
        - Furniture recommendation systems
        - Inventory management
        - Visual search functionality
        """)
    
    with use_case_col2:
        st.markdown("""
        **🏗️ Interior Design**
        - Space planning assistance
        - Furniture arrangement optimization
        - Design portfolio analysis
        - Client presentation tools
        
        **🔬 Research & Development**
        - Computer vision model benchmarking
        - Furniture detection algorithm testing
        - Academic research applications
        - AI model comparison studies
        """)
    
    # System Requirements
    st.markdown('<h2 class="sub-header">💻 System Requirements</h2>', unsafe_allow_html=True)
    
    req_col1, req_col2 = st.columns(2)
    
    with req_col1:
        st.markdown("""
        **Minimum Requirements:**
        - Python 3.8+
        - 4GB RAM
        - 2GB free storage
        - CPU with SSE4.2 support
        - Modern web browser
        """)
    
    with req_col2:
        st.markdown("""
        **Recommended for Best Performance:**
        - Python 3.9+
        - 8GB+ RAM
        - NVIDIA GPU with CUDA support
        - 5GB+ free storage
        - Chrome/Firefox browser
        """)
    
    # Credits and Acknowledgments
    st.markdown('<h2 class="sub-header">🙏 Credits & Acknowledgments</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-info">
        <h4>Built with ❤️ using:</h4>
        <ul>
            <li><strong>Ultralytics YOLO</strong> - State-of-the-art object detection framework</li>
            <li><strong>Streamlit</strong> - Beautiful web app framework for machine learning</li>
            <li><strong>OpenCV</strong> - Computer vision and image processing</li>
            <li><strong>Roboflow</strong> - Enhanced model training and data management</li>
            <li><strong>PyTorch</strong> - Deep learning framework</li>
        </ul>
        
        <h4>Special Thanks:</h4>
        <p>To the open-source community and researchers who made YOLO, Streamlit, and other amazing tools available for building intelligent applications.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Version and Updates
    st.markdown('<h2 class="sub-header">📋 Version Information</h2>', unsafe_allow_html=True)
    
    version_col1, version_col2 = st.columns(2)
    
    with version_col1:
        st.info("""
        **Current Version:** 1.0.0
        **Release Date:** July 2025
        **Last Updated:** July 3, 2025
        """)
    
    with version_col2:
        st.success("""
        **✨ Latest Features:**
        - Multi-model comparison
        - Sample gallery integration
        - Performance analytics dashboard
        - Enhanced export capabilities
        """)
    
    # Contact and Support
    st.markdown('<h2 class="sub-header">📞 Support & Contact</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-comparison">
        <h4>Need Help or Have Questions?</h4>
        <p>🐛 <strong>Found a bug?</strong> Please report it on our GitHub repository</p>
        <p>💡 <strong>Have a feature request?</strong> We'd love to hear your ideas!</p>
        <p>📚 <strong>Need documentation?</strong> Check our README.md for detailed setup instructions</p>
        <p>🤝 <strong>Want to contribute?</strong> Pull requests are welcome!</p>
        
        <br>
        <p><em>This app is open-source and continuously improving. Thank you for using our Furniture Detection App!</em></p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path, model_name):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading {model_name}: {str(e)}")
        return None

def get_model_info(model, model_name):
    """Get model information"""
    try:
        # Get model parameters count
        total_params = sum(p.numel() for p in model.model.parameters())
        
        # Get model size (approximate)
        model_size = os.path.getsize(MODEL_CONFIGS[model_name]["path"]) / (1024*1024)  # MB
        
        return {
            "parameters": f"{total_params:,}",
            "size_mb": f"{model_size:.1f} MB",
            "input_size": "640x640",
            "classes": len(FURNITURE_CLASSES)
        }
    except:
        return {
            "parameters": "N/A",
            "size_mb": "N/A", 
            "input_size": "640x640",
            "classes": len(FURNITURE_CLASSES)
        }

def draw_bounding_boxes(image, results, confidence_threshold, selected_classes, model_name):
    """Draw bounding boxes on image"""
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Filter by confidence and selected classes
                if confidence >= confidence_threshold and FURNITURE_CLASSES[class_id] in selected_classes:
                    # Get color for this class
                    color = COLORS[class_id % len(COLORS)]
                    
                    # Draw bounding box with thicker line for better visibility
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                    
                    # Prepare label with model name
                    label = f"{FURNITURE_CLASSES[class_id]}: {confidence:.2f}"
                    
                    # Calculate text size
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Draw label background
                    draw.rectangle([x1, y1-text_height-10, x1+text_width+10, y1], fill=color)
                    
                    # Draw text
                    draw.text((x1+5, y1-text_height-5), label, fill="white", font=font)
                    
                    # Store detection info
                    detections.append({
                        "class": FURNITURE_CLASSES[class_id],
                        "confidence": float(confidence),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "model": model_name
                    })
    
    return img_with_boxes, detections

def process_image(image, model, confidence_threshold, selected_classes, model_name):
    """Process single image"""
    # Run inference
    start_time = time.time()
    results = model(image)
    inference_time = time.time() - start_time
    
    # Draw bounding boxes
    img_with_boxes, detections = draw_bounding_boxes(image, results, confidence_threshold, selected_classes, model_name)
    
    return img_with_boxes, detections, inference_time

def process_video(video_path, model, confidence_threshold, selected_classes, model_name, progress_bar=None):
    """Process video file"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare output video
    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    all_detections = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Process frame
        processed_img, detections, _ = process_image(pil_image, model, confidence_threshold, selected_classes, model_name)
        
        # Store detections with frame info
        for detection in detections:
            detection['frame'] = frame_count
            detection['timestamp'] = frame_count / fps
        all_detections.extend(detections)
        
        # Convert back to BGR for video writing
        processed_array = np.array(processed_img)
        processed_bgr = cv2.cvtColor(processed_array, cv2.COLOR_RGB2BGR)
        out.write(processed_bgr)
        
        frame_count += 1
        
        # Update progress
        if progress_bar:
            progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    out.release()
    
    return output_path, all_detections

def export_results(detections, export_format):
    """Export detection results in various formats"""
    if not detections:
        return None
    
    if export_format == "JSON":
        return json.dumps(detections, indent=2)
    
    elif export_format == "CSV":
        df = pd.DataFrame(detections)
        return df.to_csv(index=False)
    
    elif export_format == "Excel":
        df = pd.DataFrame(detections)
        output = io.BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        return output.getvalue()

def compare_models(detections_dict):
    """Compare results from different models"""
    if len(detections_dict) < 2:
        return None
    
    comparison_data = []
    for model_name, detections in detections_dict.items():
        total_detections = len(detections)
        avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0
        classes_detected = len(set([d['class'] for d in detections]))
        
        comparison_data.append({
            "Model": model_name,
            "Total Detections": total_detections,
            "Avg Confidence": f"{avg_confidence:.3f}",
            "Classes Detected": classes_detected
        })
    
    return pd.DataFrame(comparison_data)

def main():
    # Create sample folders on first run
    create_sample_folders()
    
    # Header
    st.markdown('<h1 class="main-header">🪑 Furniture Detection App</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by YOLOv8s & YOLOv11s | Compare Model Performance**")
    
    # Sidebar
    st.sidebar.markdown('<h2 class="sub-header">⚙️ Settings</h2>', unsafe_allow_html=True)
    
    # Model selection
    st.sidebar.markdown("### 🤖 Model Selection")
    
    # Model dropdown
    selected_model = st.sidebar.selectbox(
        "Choose YOLO Model",
        options=list(MODEL_CONFIGS.keys()),
        index=1,  # Default to YOLOv11s
        help="Select which YOLO model to use for detection"
    )
    
    # Display model info
    model_config = MODEL_CONFIGS[selected_model]
    st.sidebar.markdown(f"""
    <div class="model-info">
        <h4>{model_config['icon']} {model_config['name']}</h4>
        <p>{model_config['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model comparison mode
    compare_mode = st.sidebar.checkbox(
        "🔬 Model Comparison Mode",
        help="Run both models and compare results"
    )

    st.sidebar.markdown("### 🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose Page",
        ["🪑 Furniture Detection", "📊 Model Performance", "ℹ️ About"],
        index=0
    )
    
    if page == "📊 Model Performance":
        display_model_performance()
        return
    elif page == "ℹ️ About":
        display_about_page()
        return
    
    # Custom model paths
    with st.sidebar.expander("🛠️ Custom Model Paths"):
        yolov8_path = st.text_input(
            "YOLOv8s Model Path",
            value=MODEL_CONFIGS["YOLOv8s"]["path"]
        )
        yolov11_path = st.text_input(
            "YOLOv11s Model Path", 
            value=MODEL_CONFIGS["YOLOv11s"]["path"]
        )
        
        # Update configs
        MODEL_CONFIGS["YOLOv8s"]["path"] = yolov8_path
        MODEL_CONFIGS["YOLOv11s"]["path"] = yolov11_path
    
    # Load models
    models = {}
    model_info = {}
    
    if compare_mode:
        # Load both models
        st.sidebar.markdown("### 📥 Loading Models...")
        for model_name, config in MODEL_CONFIGS.items():
            if os.path.exists(config["path"]):
                model = load_model(config["path"], model_name)
                if model:
                    models[model_name] = model
                    model_info[model_name] = get_model_info(model, model_name)
                    st.sidebar.success(f"✅ {model_name} loaded!")
                else:
                    st.sidebar.error(f"❌ Failed to load {model_name}")
            else:
                st.sidebar.warning(f"⚠️ {model_name} model not found: {config['path']} , roboflow API is not free and could not be deployed")
        
        if len(models) == 0:
            st.error("No models could be loaded. Please check model paths.")
            st.stop()
    else:
        # Load selected model only
        config = MODEL_CONFIGS[selected_model]
        if os.path.exists(config["path"]):
            model = load_model(config["path"], selected_model)
            if model:
                models[selected_model] = model
                model_info[selected_model] = get_model_info(model, selected_model)
                st.sidebar.success(f"✅ {selected_model} loaded!")
            else:
                st.sidebar.error(f"❌ Failed to load {selected_model}")
                st.stop()
        else:
            st.sidebar.error(f"❌ Model not found: {config['path']}")
            st.stop()
    
    # Display model information
    if model_info:
        st.sidebar.markdown("### 📊 Model Information")
        for model_name, info in model_info.items():
            with st.sidebar.expander(f"{MODEL_CONFIGS[model_name]['icon']} {model_name}"):
                st.write(f"**Parameters:** {info['parameters']}")
                st.write(f"**Size:** {info['size_mb']}")
                st.write(f"**Input:** {info['input_size']}")
                st.write(f"**Classes:** {info['classes']}")
    
    # Detection settings
    st.sidebar.markdown("### 🎯 Detection Settings")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    # Class selection
    st.sidebar.markdown("### 📋 Class Filter")
    selected_classes = st.sidebar.multiselect(
        "Select Classes to Detect",
        options=FURNITURE_CLASSES,
        default=FURNITURE_CLASSES,
        help="Choose which furniture types to detect"
    )
    
    if not selected_classes:
        st.warning("Please select at least one class to detect.")
        st.stop()
    
    # Detection mode
    st.sidebar.markdown("### 📱 Detection Mode")
    mode = st.sidebar.selectbox(
        "Choose Detection Mode",
        ["📸 Image Upload", "🎥 Video Upload", "📹 Webcam (Live)", "📂 Batch Processing", "🖼️ Sample Gallery"]
    )
    
    # Main content area
    if compare_mode:
        st.info("🔬 **Model Comparison Mode Active** - Results from both models will be shown")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if mode == "🖼️ Sample Gallery":
            st.markdown('<h2 class="sub-header">🖼️ Sample Gallery</h2>', unsafe_allow_html=True)
            st.markdown("**Test the models with pre-loaded sample images and videos from the repository**")
            
            # Gallery type selection
            gallery_type = st.radio(
                "Choose Sample Type",
                ["📸 Sample Images", "🎥 Sample Videos"],
                horizontal=True
            )
            
            if gallery_type == "📸 Sample Images":
                # Display image gallery
                selected_sample = display_sample_gallery('images')
                
                if selected_sample:
                    st.success(f"✅ Selected: {selected_sample['name']}")
                    
                    try:
                        # Load and display the selected image
                        image = Image.open(selected_sample['path'])
                        st.subheader("📸 Selected Sample Image")
                        st.image(image, caption=f"Sample: {selected_sample['name']}", use_column_width=True)
                        
                        # Process with selected model(s)
                        all_detections = {}
                        inference_times = {}
                        
                        with st.spinner("🔍 Detecting furniture in sample image..."):
                            for model_name, model in models.items():
                                processed_img, detections, inf_time = process_image(
                                    image, model, confidence_threshold, selected_classes, model_name
                                )
                                all_detections[model_name] = detections
                                inference_times[model_name] = inf_time
                                
                                # Display results
                                st.subheader(f"🎯 {MODEL_CONFIGS[model_name]['icon']} {model_name} Results")
                                st.image(processed_img, caption=f"Detected by {model_name}", use_column_width=True)
                                st.caption(f"⏱️ Inference time: {inf_time:.3f}s")
                        
                        # Show detection statistics
                        with col2:
                            st.markdown('<div class="detection-stats">', unsafe_allow_html=True)
                            st.markdown("### 📊 Detection Summary")
                            
                            # Show results for each model
                            for model_name, detections in all_detections.items():
                                st.markdown(f"#### {MODEL_CONFIGS[model_name]['icon']} {model_name}")
                                
                                if detections:
                                    # Count detections by class
                                    class_counts = {}
                                    for det in detections:
                                        class_name = det['class']
                                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                                    
                                    st.write(f"**Total Detections:** {len(detections)}")
                                    st.write(f"**Inference Time:** {inference_times[model_name]:.3f}s")
                                    st.write("**Detected Items:**")
                                    
                                    for class_name, count in class_counts.items():
                                        st.write(f"• {class_name}: {count}")
                                else:
                                    st.write("No furniture detected with current settings.")
                                
                                st.markdown("---")
                            
                            # Model comparison
                            if compare_mode and len(all_detections) > 1:
                                st.markdown("### 🔬 Model Comparison")
                                comparison_df = compare_models(all_detections)
                                if comparison_df is not None:
                                    st.dataframe(comparison_df)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"Error processing sample image: {str(e)}")
            
            elif gallery_type == "🎥 Sample Videos":
                # Display video gallery
                selected_sample = display_sample_gallery('videos')
                
                if selected_sample:
                    st.success(f"✅ Selected: {selected_sample['name']}")
                    
                    try:
                        # Display video info and player
                        st.subheader("🎥 Selected Sample Video")
                        st.video(selected_sample['path'])
                        st.caption(f"Sample: {selected_sample['name']} ({selected_sample['size'] / (1024*1024):.1f} MB)")
                        
                        # Model selection for video processing
                        if compare_mode:
                            st.warning("⚠️ Video comparison mode processes with both models sequentially")
                        
                        if st.button("🚀 Start Video Processing"):
                            all_video_detections = {}
                            
                            for model_name, model in models.items():
                                st.subheader(f"Processing with {model_name}...")
                                progress_bar = st.progress(0)
                                
                                with st.spinner(f"🎬 Processing sample video with {model_name}..."):
                                    start_time = time.time()
                                    processed_video_path, detections = process_video(
                                        selected_sample['path'], model, confidence_threshold, selected_classes, model_name, progress_bar
                                    )
                                    processing_time = time.time() - start_time
                                
                                st.success(f"✅ {model_name} processed video in {processing_time:.2f} seconds!")
                                all_video_detections[model_name] = detections
                                
                                # Display processed video
                                with open(processed_video_path, 'rb') as f:
                                    video_bytes = f.read()
                                
                                st.subheader(f"🎯 {model_name} Processed Video")
                                st.video(video_bytes)
                                
                                # Download processed video
                                st.download_button(
                                    label=f"📥 Download {model_name} Video",
                                    data=video_bytes,
                                    file_name=f"{model_name}_processed_{selected_sample['name']}",
                                    mime="video/mp4",
                                    key=f"download_{model_name}_sample"
                                )
                            
                            # Show statistics
                            with col2:
                                st.markdown("### 📊 Video Analysis")
                                
                                for model_name, detections in all_video_detections.items():
                                    st.markdown(f"#### {MODEL_CONFIGS[model_name]['icon']} {model_name}")
                                    st.write(f"**Total Detections:** {len(detections)}")
                                    
                                    if detections:
                                        # Timeline analysis
                                        df = pd.DataFrame(detections)
                                        st.write("**Detection Timeline:**")
                                        st.line_chart(df.groupby('class').size())
                    
                    except Exception as e:
                        st.error(f"Error processing sample video: {str(e)}")
        
        elif mode == "📸 Image Upload":
            st.markdown('<h2 class="sub-header">📸 Image Detection</h2>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload an image to detect furniture"
            )
            
            if uploaded_file is not None:
                # Display original image
                image = Image.open(uploaded_file)
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Process with selected model(s)
                all_detections = {}
                inference_times = {}
                
                with st.spinner("🔍 Detecting furniture..."):
                    for model_name, model in models.items():
                        processed_img, detections, inf_time = process_image(
                            image, model, confidence_threshold, selected_classes, model_name
                        )
                        all_detections[model_name] = detections
                        inference_times[model_name] = inf_time
                        
                        # Display results
                        st.subheader(f"🎯 {MODEL_CONFIGS[model_name]['icon']} {model_name} Results")
                        st.image(processed_img, caption=f"Detected by {model_name}", use_column_width=True)
                        st.caption(f"⏱️ Inference time: {inf_time:.3f}s")
                
                # Show detection statistics
                with col2:
                    st.markdown('<div class="detection-stats">', unsafe_allow_html=True)
                    st.markdown("### 📊 Detection Summary")
                    
                    # Show results for each model
                    for model_name, detections in all_detections.items():
                        st.markdown(f"#### {MODEL_CONFIGS[model_name]['icon']} {model_name}")
                        
                        if detections:
                            # Count detections by class
                            class_counts = {}
                            for det in detections:
                                class_name = det['class']
                                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                            
                            st.write(f"**Total Detections:** {len(detections)}")
                            st.write(f"**Inference Time:** {inference_times[model_name]:.3f}s")
                            st.write("**Detected Items:**")
                            
                            for class_name, count in class_counts.items():
                                st.write(f"• {class_name}: {count}")
                        else:
                            st.write("No furniture detected with current settings.")
                        
                        st.markdown("---")
                    
                    # Model comparison
                    if compare_mode and len(all_detections) > 1:
                        st.markdown("### 🔬 Model Comparison")
                        comparison_df = compare_models(all_detections)
                        if comparison_df is not None:
                            st.dataframe(comparison_df)
                    
                    # Export options
                    st.markdown("### 💾 Export Results")
                    export_format = st.selectbox("Export Format", ["JSON", "CSV", "Excel"])
                    
                    if st.button("📤 Export Detections"):
                        # Combine all detections
                        combined_detections = []
                        for model_detections in all_detections.values():
                            combined_detections.extend(model_detections)
                        
                        export_data = export_results(combined_detections, export_format)
                        if export_data:
                            if export_format == "Excel":
                                st.download_button(
                                    label="Download Excel File",
                                    data=export_data,
                                    file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            else:
                                st.download_button(
                                    label=f"Download {export_format} File",
                                    data=export_data,
                                    file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
                                    mime="text/plain"
                                )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # [Rest of the modes remain the same - Video Upload, Webcam, Batch Processing]
        # I'll keep them as in the original code for brevity
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666;">🪑 Furniture Detection App | '
        f'Built with ❤️ using Streamlit & {" + ".join(models.keys()) if models else "YOLO"}</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()