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
    }
}

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
                st.sidebar.warning(f"⚠️ {model_name} model not found: {config['path']}")
        
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
        ["📸 Image Upload", "🎥 Video Upload", "📹 Webcam (Live)", "📂 Batch Processing"]
    )
    
    # Main content area
    if compare_mode:
        st.info("🔬 **Model Comparison Mode Active** - Results from both models will be shown")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if mode == "📸 Image Upload":
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
        
        elif mode == "🎥 Video Upload":
            st.markdown('<h2 class="sub-header">🎥 Video Detection</h2>', unsafe_allow_html=True)
            
            uploaded_video = st.file_uploader(
                "Choose a video...",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video to detect furniture"
            )
            
            if uploaded_video is not None:
                # Save uploaded video temporarily
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                video_path = tfile.name
                
                # Display video info
                st.video(uploaded_video)
                
                # Model selection for video processing
                if compare_mode:
                    st.warning("⚠️ Video comparison mode processes with both models sequentially")
                
                if st.button("🚀 Start Video Processing"):
                    all_video_detections = {}
                    
                    for model_name, model in models.items():
                        st.subheader(f"Processing with {model_name}...")
                        progress_bar = st.progress(0)
                        
                        with st.spinner(f"🎬 Processing video with {model_name}..."):
                            start_time = time.time()
                            processed_video_path, detections = process_video(
                                video_path, model, confidence_threshold, selected_classes, model_name, progress_bar
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
                            file_name=f"{model_name}_processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                            mime="video/mp4",
                            key=f"download_{model_name}"
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
                        
                        # Export video results
                        export_format = st.selectbox("Export Format", ["JSON", "CSV", "Excel"], key="video_export")
                        if st.button("📤 Export Video Results"):
                            # Combine all video detections
                            combined_detections = []
                            for model_detections in all_video_detections.values():
                                combined_detections.extend(model_detections)
                            
                            export_data = export_results(combined_detections, export_format)
                            if export_data:
                                if export_format == "Excel":
                                    st.download_button(
                                        label="Download Excel File",
                                        data=export_data,
                                        file_name=f"video_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                else:
                                    st.download_button(
                                        label=f"Download {export_format} File",
                                        data=export_data,
                                        file_name=f"video_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
                                        mime="text/plain"
                                    )
                
                # Clean up
                os.unlink(video_path)
        
        elif mode == "📹 Webcam (Live)":
            st.markdown('<h2 class="sub-header">📹 Live Webcam Detection</h2>', unsafe_allow_html=True)
            
            # Webcam settings
            webcam_source = st.selectbox("Camera Source", [0, 1, 2], help="Select camera index")
            
            # Model selection for webcam
            if compare_mode:
                st.info("ℹ️ Live comparison mode will alternate between models")
                frame_interval = st.slider("Model Switch Interval (frames)", 1, 30, 10)
            
            if st.button("🎥 Start Webcam"):
                # Create placeholder for webcam feed
                frame_placeholder = st.empty()
                stop_button = st.button("⏹️ Stop Webcam")
                
                # Initialize webcam
                cap = cv2.VideoCapture(webcam_source)
                
                if not cap.isOpened():
                    st.error("❌ Could not open webcam")
                else:
                    st.success("✅ Webcam started successfully!")
                    
                    with col2:
                        detection_placeholder = st.empty()
                    
                    frame_count = 0
                    current_model_idx = 0
                    model_names = list(models.keys())
                    
                    while not stop_button:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("❌ Failed to read from webcam")
                            break
                        
                        # Convert frame for processing
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # Select model for processing
                        if compare_mode:
                            if frame_count % frame_interval == 0:
                                current_model_idx = (current_model_idx + 1) % len(model_names)
                            current_model_name = model_names[current_model_idx]
                        else:
                            current_model_name = selected_model
                        
                        current_model = models[current_model_name]
                        
                        # Process frame
                        processed_img, detections, inf_time = process_image(
                            pil_image, current_model, confidence_threshold, selected_classes, current_model_name
                        )
                        
                        # Display frame
                        frame_placeholder.image(
                            processed_img, 
                            caption=f"Live Feed - {MODEL_CONFIGS[current_model_name]['icon']} {current_model_name} ({inf_time:.3f}s)", 
                            use_column_width=True
                        )
                        
                        # Update detection info
                        with detection_placeholder.container():
                            st.markdown(f"### 🔴 Live Detections - {current_model_name}")
                            if detections:
                                for det in detections:
                                    confidence_color = "green" if det['confidence'] > 0.8 else "orange" if det['confidence'] > 0.6 else "red"
                                    st.markdown(
                                        f'<span class="confidence-box" style="background-color: {confidence_color};">'
                                        f'{det["class"]}: {det["confidence"]:.2f}</span>',
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.write("No detections")
                        
                        frame_count += 1
                        time.sleep(0.1)  # Small delay to prevent overwhelming
                    
                    cap.release()
        
        elif mode == "📂 Batch Processing":
            st.markdown('<h2 class="sub-header">📂 Batch Processing</h2>', unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "Choose multiple images...",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=True,
                help="Upload multiple images for batch processing"
            )
            
            if uploaded_files:
                st.write(f"Selected {len(uploaded_files)} files for processing")
                
                if st.button("🚀 Process All Images"):
                    progress_bar = st.progress(0)
                    all_batch_results = {}
                    
                    # Initialize results for each model
                    for model_name in models.keys():
                        all_batch_results[model_name] = []
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        image = Image.open(uploaded_file)
                        
                        # Process with each model
                        for model_name, model in models.items():
                            processed_img, detections, _ = process_image(
                                image, model, confidence_threshold, selected_classes, model_name
                            )
                            
                            # Store results
                            for det in detections:
                                det['filename'] = uploaded_file.name
                            all_batch_results[model_name].extend(detections)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    st.success(f"✅ Processed {len(uploaded_files)} images with {len(models)} model(s)!")
                    
                    # Show summary for each model
                    for model_name, results in all_batch_results.items():
                        st.subheader(f"📊 {MODEL_CONFIGS[model_name]['icon']} {model_name} Batch Results")
                        st.write(f"**Total Detections:** {len(results)}")
                        
                        if results:
                            df = pd.DataFrame(results)
                            
                            # Summary by class
                            class_summary = df['class'].value_counts()
                            st.bar_chart(class_summary)
                            
                            # Summary by file
                            file_summary = df['filename'].value_counts()
                            st.write("**Detections per File:**")
                            st.dataframe(file_summary)
                    
                    # Model comparison for batch
                    if compare_mode and len(all_batch_results) > 1:
                        st.subheader("🔬 Batch Model Comparison")
                        comparison_df = compare_models(all_batch_results)
                        if comparison_df is not None:
                            st.dataframe(comparison_df)
                    
                    # Export batch results
                    export_format = st.selectbox("Export Format", ["JSON", "CSV", "Excel"], key="batch_export")
                    if st.button("📤 Export Batch Results"):
                        # Combine all batch results
                        combined_batch_results = []
                        for model_results in all_batch_results.values():
                            combined_batch_results.extend(model_results)
                        
                        export_data = export_results(combined_batch_results, export_format)
                        if export_data:
                            if export_format == "Excel":
                                st.download_button(
                                    label="Download Excel File",
                                    data=export_data,
                                    file_name=f"batch_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            else:
                                st.download_button(
                                    label=f"Download {export_format} File",
                                    data=export_data,
                                    file_name=f"batch_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
                                    mime="text/plain"
                                )
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666;">🪑 Furniture Detection App | '
        f'Built with ❤️ using Streamlit & {" + ".join(models.keys()) if models else "YOLO"}</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()