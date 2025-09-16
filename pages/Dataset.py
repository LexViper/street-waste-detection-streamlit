"""
Dataset Statistics and Management Page
View dataset information and manage training data
"""

import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Dataset - Clean City",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.dataset-header {
    background: linear-gradient(90deg, #4ECDC4 0%, #44A08D 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.stat-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 1rem 0;
    text-align: center;
}

.category-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

def get_dataset_statistics():
    """Get comprehensive dataset statistics"""
    stats = {
        'total_images': 0,
        'train_images': 0,
        'val_images': 0,
        'categories': {'plastic': 0, 'paper': 0, 'organic': 0},
        'dataset_ready': False
    }
    
    # Check if dataset exists
    if not os.path.exists('dataset'):
        return stats
    
    # Count training images
    train_path = 'dataset/images/train'
    if os.path.exists(train_path):
        train_files = [f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        stats['train_images'] = len(train_files)
        
        # Count by category
        for file in train_files:
            if 'plastic' in file.lower():
                stats['categories']['plastic'] += 1
            elif 'paper' in file.lower() or 'cardboard' in file.lower():
                stats['categories']['paper'] += 1
            elif 'organic' in file.lower():
                stats['categories']['organic'] += 1
    
    # Count validation images
    val_path = 'dataset/images/val'
    if os.path.exists(val_path):
        val_files = [f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        stats['val_images'] = len(val_files)
    
    stats['total_images'] = stats['train_images'] + stats['val_images']
    stats['dataset_ready'] = stats['total_images'] > 0
    
    return stats

def display_sample_images(image_dir, num_samples=6):
    """Display sample images from dataset"""
    if not os.path.exists(image_dir):
        st.warning(f"Directory not found: {image_dir}")
        return
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        st.info("No images found in this directory")
        return
    
    # Select random samples
    np.random.seed(42)
    selected_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
    
    cols = st.columns(3)
    for i, img_file in enumerate(selected_files):
        with cols[i % 3]:
            try:
                img_path = os.path.join(image_dir, img_file)
                image = Image.open(img_path)
                st.image(image, caption=img_file, use_column_width=True)
            except Exception as e:
                st.error(f"Error loading {img_file}: {e}")

def main():
    """Main dataset page function"""
    
    # Header
    st.markdown("""
    <div class="dataset-header">
        <h1>📊 Dataset Management</h1>
        <p>View and manage your waste detection training data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get dataset statistics
    stats = get_dataset_statistics()
    
    # Dataset Overview
    st.markdown("## 📈 Dataset Overview")
    
    if not stats['dataset_ready']:
        st.error("❌ No dataset found!")
        st.info("""
        **To set up your dataset:**
        1. Download the Kaggle Garbage Classification v2 dataset
        2. Extract it to the project folder
        3. Run: `python download_dataset.py` and choose option 4
        4. Wait for conversion to complete
        """)
        return
    
    # Statistics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h2>{stats['total_images']:,}</h2>
            <p>Total Images</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h2>{stats['train_images']:,}</h2>
            <p>Training Images</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <h2>{stats['val_images']:,}</h2>
            <p>Validation Images</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        train_val_ratio = stats['train_images'] / stats['val_images'] if stats['val_images'] > 0 else 0
        st.markdown(f"""
        <div class="stat-card">
            <h2>{train_val_ratio:.1f}:1</h2>
            <p>Train:Val Ratio</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Category Distribution
    st.markdown("## 🗂️ Category Distribution")
    
    col5, col6 = st.columns([2, 1])
    
    with col5:
        # Pie chart
        categories = list(stats['categories'].keys())
        counts = list(stats['categories'].values())
        colors = ['#FF6B6B', '#FFE66D', '#4ECDC4']
        
        fig = go.Figure(data=[go.Pie(
            labels=categories,
            values=counts,
            marker_colors=colors,
            hole=0.4
        )])
        
        fig.update_layout(
            title="Dataset Category Distribution",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col6:
        # Category stats
        for category, count in stats['categories'].items():
            percentage = (count / stats['total_images']) * 100 if stats['total_images'] > 0 else 0
            st.markdown(f"""
            <div class="category-card">
                <h4>{category.capitalize()}</h4>
                <p>{count:,} images ({percentage:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Sample Images
    st.markdown("## 🖼️ Sample Images")
    
    tab1, tab2 = st.tabs(["Training Samples", "Validation Samples"])
    
    with tab1:
        st.markdown("### Training Dataset Samples")
        display_sample_images('dataset/images/train')
    
    with tab2:
        st.markdown("### Validation Dataset Samples")
        display_sample_images('dataset/images/val')
    
    # Dataset Health Check
    st.markdown("## 🏥 Dataset Health Check")
    
    health_checks = []
    
    # Check minimum images per category
    min_images_per_category = 100
    for category, count in stats['categories'].items():
        if count < min_images_per_category:
            health_checks.append(f"⚠️ {category.capitalize()} has only {count} images (recommended: {min_images_per_category}+)")
        else:
            health_checks.append(f"✅ {category.capitalize()} has sufficient images ({count})")
    
    # Check train/val ratio
    if train_val_ratio < 3 or train_val_ratio > 5:
        health_checks.append(f"⚠️ Train/Val ratio is {train_val_ratio:.1f}:1 (recommended: 4:1)")
    else:
        health_checks.append(f"✅ Good Train/Val ratio ({train_val_ratio:.1f}:1)")
    
    # Check total dataset size
    if stats['total_images'] < 1000:
        health_checks.append(f"⚠️ Small dataset ({stats['total_images']} images). Consider adding more data.")
    else:
        health_checks.append(f"✅ Good dataset size ({stats['total_images']:,} images)")
    
    for check in health_checks:
        if "✅" in check:
            st.success(check)
        else:
            st.warning(check)
    
    # Dataset Actions
    st.markdown("## ⚡ Dataset Actions")
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        if st.button("🔄 Refresh Statistics"):
            st.rerun()
    
    with col8:
        if st.button("📊 Generate Report"):
            st.info("Dataset report generation would be implemented here")
    
    with col9:
        if st.button("🧹 Clean Dataset"):
            st.info("Dataset cleaning tools would be implemented here")

if __name__ == "__main__":
    main()
