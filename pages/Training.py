"""
Training Page for Clean City Waste Detection Application
Monitor training progress and manage models
"""

import streamlit as st
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import subprocess
import time

# Page configuration
st.set_page_config(
    page_title="Training - Clean City",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.training-header {
    background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.status-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 1rem 0;
    border-left: 4px solid #4ECDC4;
}

.metric-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem;
}

.progress-container {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def check_dataset_status():
    """Check if dataset is ready for training"""
    dataset_yaml = "dataset.yaml"
    dataset_images = "dataset/images"
    
    if not os.path.exists(dataset_yaml):
        return False, "Dataset configuration not found"
    
    if not os.path.exists(dataset_images):
        return False, "Dataset images not found"
    
    # Count images
    train_path = "dataset/images/train"
    val_path = "dataset/images/val"
    
    train_count = 0
    val_count = 0
    
    if os.path.exists(train_path):
        train_count = len([f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if os.path.exists(val_path):
        val_count = len([f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if train_count == 0 or val_count == 0:
        return False, f"Insufficient images (Train: {train_count}, Val: {val_count})"
    
    return True, f"Ready (Train: {train_count}, Val: {val_count})"

def check_training_status():
    """Check if training is currently running"""
    # Check for training results
    results_path = "training_results"
    if os.path.exists(results_path):
        # Look for recent training files
        for item in os.listdir(results_path):
            item_path = os.path.join(results_path, item)
            if os.path.isdir(item_path) and "waste_detection_training" in item:
                return True, item_path
    
    return False, None

def load_training_summary():
    """Load training summary if available"""
    summary_file = "training_results/training_summary.json"
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            return json.load(f)
    return None

def display_training_metrics(summary):
    """Display training metrics in a nice format"""
    if not summary:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{summary['metrics']['map50']:.3f}</h3>
            <p>mAP@0.5</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{summary['metrics']['precision']:.3f}</h3>
            <p>Precision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{summary['metrics']['recall']:.3f}</h3>
            <p>Recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{summary['training_time_hours']:.1f}h</h3>
            <p>Training Time</p>
        </div>
        """, unsafe_allow_html=True)

def create_training_progress_chart():
    """Create a mock training progress chart"""
    # This would be replaced with real training logs in production
    epochs = list(range(1, 51))
    train_loss = [0.8 - (i * 0.01) + (0.1 * (i % 10) / 10) for i in epochs]
    val_loss = [0.85 - (i * 0.009) + (0.12 * (i % 8) / 8) for i in epochs]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss,
        mode='lines',
        name='Training Loss',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs, y=val_loss,
        mode='lines',
        name='Validation Loss',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    fig.update_layout(
        title='Training Progress',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        height=400,
        margin=dict(t=50, b=40, l=40, r=40)
    )
    
    return fig

def main():
    """Main training page function"""
    
    # Header
    st.markdown("""
    <div class="training-header">
        <h1>🎯 Model Training Center</h1>
        <p>Train and manage your waste detection models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check dataset status
    dataset_ready, dataset_status = check_dataset_status()
    training_active, training_path = check_training_status()
    
    # Status Overview
    st.markdown("## 📊 Training Status Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        status_color = "🟢" if dataset_ready else "🔴"
        st.markdown(f"""
        <div class="status-card">
            <h4>{status_color} Dataset Status</h4>
            <p>{dataset_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        training_color = "🟡" if training_active else "⚪"
        training_text = "Training in progress" if training_active else "No active training"
        st.markdown(f"""
        <div class="status-card">
            <h4>{training_color} Training Status</h4>
            <p>{training_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset Preparation Section
    st.markdown("---")
    st.markdown("## 📁 Dataset Preparation")
    
    if not dataset_ready:
        st.warning("⚠️ Dataset not ready for training!")
        
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("🔄 Convert Kaggle Dataset", type="primary"):
                with st.spinner("Converting dataset..."):
                    # Run dataset conversion
                    result = subprocess.run(
                        ["python3", "download_dataset.py"],
                        input="4\n",
                        text=True,
                        capture_output=True
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ Dataset converted successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Conversion failed: {result.stderr}")
        
        with col4:
            st.info("""
            **Steps to prepare dataset:**
            1. Download Kaggle dataset
            2. Extract to project folder
            3. Click 'Convert Kaggle Dataset'
            4. Wait for conversion to complete
            """)
    
    else:
        st.success("✅ Dataset is ready for training!")
        
        # Show dataset statistics
        with st.expander("📊 Dataset Statistics"):
            train_count = len([f for f in os.listdir("dataset/images/train") if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            val_count = len([f for f in os.listdir("dataset/images/val") if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            col5, col6, col7 = st.columns(3)
            with col5:
                st.metric("Training Images", train_count)
            with col6:
                st.metric("Validation Images", val_count)
            with col7:
                st.metric("Total Images", train_count + val_count)
    
    # Training Section
    st.markdown("---")
    st.markdown("## 🚀 Model Training")
    
    if dataset_ready and not training_active:
        st.markdown("### Training Configuration")
        
        col8, col9, col10 = st.columns(3)
        
        with col8:
            epochs = st.number_input("Epochs", min_value=10, max_value=200, value=50)
        
        with col9:
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)
        
        with col10:
            img_size = st.selectbox("Image Size", [416, 640, 832], index=1)
        
        # Estimated training time
        estimated_time = epochs * 2  # rough estimate: 2 minutes per epoch
        st.info(f"⏱️ Estimated training time: {estimated_time//60}h {estimated_time%60}m")
        
        # Start training button
        if st.button("🎯 Start Training", type="primary"):
            st.warning("🚧 Training would start here. For demo purposes, this shows the UI flow.")
            
            # In production, this would start the actual training
            with st.spinner("Starting training..."):
                time.sleep(2)
                st.success("Training started! Check the progress below.")
    
    elif training_active:
        st.info("🔄 Training is currently in progress...")
        
        # Show training progress (mock)
        progress_chart = create_training_progress_chart()
        st.plotly_chart(progress_chart, use_container_width=True)
        
        # Training logs (mock)
        with st.expander("📝 Training Logs"):
            st.code("""
Epoch 45/50: 100%|██████████| 125/125 [02:15<00:00,  1.08s/it]
      Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
        all       2500       3750      0.847      0.792      0.823     0.567
    plastic       2500       1250      0.856      0.812      0.845     0.592
      paper       2500       1200      0.834      0.778      0.808     0.548
    organic       2500       1300      0.851      0.786      0.816     0.561

Validation results saved to training_results/waste_detection_training/
            """)
    
    # Previous Training Results
    st.markdown("---")
    st.markdown("## 📈 Training Results")
    
    summary = load_training_summary()
    
    if summary:
        st.success("✅ Found previous training results!")
        
        # Display metrics
        display_training_metrics(summary)
        
        # Model info
        st.markdown("### 🤖 Trained Model")
        
        model_path = summary.get('model_path', 'yolo_model/yolov8_waste_trained.pt')
        model_exists = os.path.exists(model_path)
        
        col11, col12 = st.columns(2)
        
        with col11:
            status_icon = "✅" if model_exists else "❌"
            st.markdown(f"""
            **Model Status:** {status_icon} {'Available' if model_exists else 'Not Found'}
            
            **Model Path:** `{model_path}`
            
            **Training Date:** {summary.get('timestamp', 'Unknown')[:10]}
            """)
        
        with col12:
            if model_exists:
                st.success("🎉 Model ready for use in main app!")
                if st.button("🔄 Restart Streamlit App"):
                    st.info("Please restart the main Streamlit app to use the trained model")
            else:
                st.error("❌ Model file not found. Please retrain.")
        
        # Detailed metrics
        with st.expander("📊 Detailed Metrics"):
            metrics_data = summary.get('metrics', {})
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95'],
                    y=[
                        metrics_data.get('precision', 0),
                        metrics_data.get('recall', 0),
                        metrics_data.get('map50', 0),
                        metrics_data.get('map50_95', 0)
                    ],
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                )
            ])
            
            fig.update_layout(
                title='Model Performance Metrics',
                yaxis_title='Score',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("ℹ️ No training results found. Train a model to see results here.")
    
    # Quick Actions
    st.markdown("---")
    st.markdown("## ⚡ Quick Actions")
    
    col13, col14, col15 = st.columns(3)
    
    with col13:
        if st.button("📊 View Dataset"):
            st.info("Dataset viewer would open here")
    
    with col14:
        if st.button("🔄 Reset Training"):
            st.warning("This would reset all training progress")
    
    with col15:
        if st.button("💾 Export Model"):
            st.info("Model export functionality would be here")

if __name__ == "__main__":
    main()
