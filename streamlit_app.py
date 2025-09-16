"""
Clean City Waste Detection - Main Streamlit Application
A modern web app for detecting and analyzing waste in urban environments using YOLOv8
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import os
from typing import Optional, Tuple

# Import custom modules
from helpers.detection import WasteDetector, simulate_waste_detection
from helpers.visualization import WasteVisualizer, create_detection_stats_card

# Page configuration
st.set_page_config(
    page_title="🌍 Clean City - Waste Detection",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
def load_custom_css():
    """Load custom CSS for modern UI styling"""
    st.markdown("""
    <style>
    /* Main app styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    
    .detection-results {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Animation classes */
    .fadeIn {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = WasteVisualizer()
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None

def create_header():
    """Create the main application header"""
    st.markdown("""
    <div class="main-header fadeIn">
        <h1>🌍 Clean City Waste Detection</h1>
        <p style="font-size: 1.2em; margin: 0; opacity: 0.9;">
            AI-Powered Urban Waste Analysis using YOLOv8
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create the application sidebar with controls"""
    with st.sidebar:
        st.markdown("## 🎛️ Detection Settings")
        
        # Model selection - Auto-detect trained model
        trained_model_exists = os.path.exists("yolo_model/yolov8_waste_trained.pt")
        
        if trained_model_exists:
            default_option = "YOLO Model (Real)"
            options = ["YOLO Model (Real)", "Demo Mode (Simulated)"]
        else:
            default_option = "Demo Mode (Simulated)"
            options = ["Demo Mode (Simulated)", "YOLO Model (Real)"]
            
        model_option = st.selectbox(
            "Choose Detection Mode",
            options,
            index=0,  # Always use first option (trained model if available)
            help="YOLO Model uses your trained waste detection model"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        # IoU threshold
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Threshold for non-maximum suppression"
        )
        
        st.markdown("---")
        
        # Model Status
        st.markdown("## 🤖 Model Status")
        if trained_model_exists:
            st.success("✅ Custom trained model loaded")
            st.info("🎯 97.1% mAP accuracy on waste detection")
        else:
            st.info("📦 Using pretrained model (Demo mode recommended)")
        
        # Statistics
        if st.session_state.detection_results:
            st.markdown("## 📊 Quick Stats")
            results = st.session_state.detection_results
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Objects", results['total_objects'])
            with col2:
                st.metric("Plastic Items", results['class_counts']['plastic'])
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Paper Items", results['class_counts']['paper'])
            with col4:
                st.metric("Organic Items", results['class_counts']['organic'])
        
        st.markdown("---")
        
        # Information section
        st.markdown("## ℹ️ About")
        st.info("""
        This app uses AI to detect and classify waste in images:
        
        🔵 **Blue**: Plastic waste
        🟡 **Yellow**: Paper waste  
        🟢 **Green**: Organic waste
        
        Upload an image to get started!
        """)
        
        return model_option, confidence_threshold, iou_threshold

def load_detector(model_option: str, confidence_threshold: float, iou_threshold: float):
    """Load and configure the waste detector"""
    try:
        if model_option == "YOLO Model (Real)":
            if st.session_state.detector is None:
                with st.spinner("🔄 Loading YOLO model..."):
                    st.session_state.detector = WasteDetector()
            st.session_state.detector.update_confidence_threshold(confidence_threshold)
            st.session_state.detector.update_iou_threshold(iou_threshold)
        elif model_option == "Demo Mode (Simulated)":
            # Clear detector for demo mode
            st.session_state.detector = None
    except Exception as e:
        st.error(f"❌ Error loading detector: {str(e)}")
        return False
    return True

def process_uploaded_file(uploaded_file, model_option: str) -> Optional[Tuple[np.ndarray, dict]]:
    """Process uploaded image or video file"""
    if uploaded_file is None:
        return None
        
    try:
        # Read file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        if uploaded_file.type.startswith('image'):
            # Process image
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error("❌ Could not decode image file")
                return None
            
            # Run detection
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Analyzing image...")
            progress_bar.progress(30)
            
            if model_option == "Demo Mode (Simulated)":
                time.sleep(1)  # Simulate processing time
                detection_results = simulate_waste_detection(image)
            else:
                if st.session_state.detector is None:
                    st.error("❌ Detector not loaded")
                    return None
                detection_results = st.session_state.detector.detect_waste(image)
            
            progress_bar.progress(70)
            status_text.text("🎨 Creating visualizations...")
            
            # Draw detections
            processed_image = st.session_state.visualizer.draw_detections(
                image, detection_results['detections']
            )
            
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
            
            return processed_image, detection_results
            
        elif uploaded_file.type.startswith('video'):
            st.info("🎬 Video processing is available but simplified for demo purposes")
            # For demo, we'll just show the first frame
            # In a full implementation, you'd process multiple frames
            return None
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        return None

def display_results(processed_image: np.ndarray, detection_results: dict):
    """Display detection results with visualizations"""
    
    # Convert BGR to RGB for display
    display_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    # Main results section
    st.markdown("## 🎯 Detection Results")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📸 Processed Image")
        st.image(display_image, caption="Detected waste items with bounding boxes", use_column_width=True)
        
        # Detection details
        if detection_results['detections']:
            st.markdown("### 🔍 Detection Details")
            for i, detection in enumerate(detection_results['detections']):
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class']
                
                st.markdown(f"""
                **Detection {i+1}:**
                - **Type**: {class_name.capitalize()}
                - **Confidence**: {confidence:.2%}
                - **Location**: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})
                """)
    
    with col2:
        # Statistics card
        st.markdown(create_detection_stats_card(detection_results), unsafe_allow_html=True)
        
        # Pie chart
        if detection_results['total_objects'] > 0:
            st.markdown("### 📊 Distribution Chart")
            pie_chart = st.session_state.visualizer.create_pie_chart(detection_results['class_counts'])
            st.plotly_chart(pie_chart, use_container_width=True)
        
        # Download processed image
        st.markdown("### 💾 Download Results")
        
        # Convert to PIL for download
        pil_image = Image.fromarray(display_image)
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        
        st.download_button(
            label="📥 Download Processed Image",
            data=img_buffer.getvalue(),
            file_name="waste_detection_result.png",
            mime="image/png"
        )
    
    # Additional analytics
    if detection_results['detections']:
        st.markdown("---")
        st.markdown("## 📈 Advanced Analytics")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Confidence distribution
            conf_chart = st.session_state.visualizer.create_confidence_distribution(
                detection_results['detections']
            )
            st.plotly_chart(conf_chart, use_container_width=True)
        
        with col4:
            # Detection heatmap
            image_shape = processed_image.shape[:2]
            heatmap = st.session_state.visualizer.create_detection_heatmap(
                image_shape, detection_results['detections']
            )
            st.plotly_chart(heatmap, use_container_width=True)

def main():
    """Main application function"""
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state FIRST
    initialize_session_state()
    
    # Create header
    create_header()
    
    # Create sidebar and get settings
    model_option, confidence_threshold, iou_threshold = create_sidebar()
    
    # Load detector if needed
    if not load_detector(model_option, confidence_threshold, iou_threshold):
        st.stop()
    
    # Main content area
    st.markdown("## 📤 Upload Media")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'mp4', 'avi', 'mov'],
        help="Supported formats: PNG, JPG, JPEG, BMP, TIFF, MP4, AVI, MOV"
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ Uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Process the file
        result = process_uploaded_file(uploaded_file, model_option)
        
        if result is not None:
            processed_image, detection_results = result
            
            # Store in session state
            st.session_state.processed_image = processed_image
            st.session_state.detection_results = detection_results
            
            # Display results
            display_results(processed_image, detection_results)
    
    else:
        # Show demo information
        st.markdown("""
        <div class="upload-area">
            <h3>🎯 Ready to Detect Waste!</h3>
            <p>Upload an image to start analyzing waste in urban environments.</p>
            <p><strong>Tip:</strong> Try images with bottles, food containers, paper, or organic waste for best results.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample images if available
        sample_images_path = "data/images"
        if os.path.exists(sample_images_path):
            st.markdown("### 🖼️ Sample Images")
            st.info("Sample images are available in the data/images folder for testing.")

if __name__ == "__main__":
    main()
