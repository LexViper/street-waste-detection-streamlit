"""
About Page for Clean City Waste Detection Application
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="About - Clean City",
    page_icon="ℹ️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.about-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.feature-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin: 1rem 0;
    border-left: 4px solid #667eea;
}

.tech-badge {
    display: inline-block;
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    margin: 0.25rem;
    font-weight: bold;
}

.stats-container {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="about-header">
        <h1>🌍 About Clean City Project</h1>
        <p style="font-size: 1.2em; margin: 0;">
            AI-Powered Solution for Urban Waste Management
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mission Statement
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 Our Mission
        
        The Clean City Project leverages cutting-edge artificial intelligence to revolutionize urban waste management. 
        Our goal is to create cleaner, more sustainable cities by providing automated waste detection and analysis tools 
        that help municipalities, environmental organizations, and citizens identify and address waste problems efficiently.
        
        ### 🔍 What We Do
        
        Our AI system can:
        - **Detect and classify** different types of waste in images and videos
        - **Analyze waste distribution** patterns in urban environments  
        - **Generate insights** to help optimize cleaning operations
        - **Monitor progress** in waste reduction initiatives
        - **Support decision-making** for urban planning and environmental policy
        """)
        
        # Technology Stack
        st.markdown("## 🛠️ Technology Stack")
        
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            st.markdown("""
            **AI & Machine Learning:**
            - YOLOv8 (Ultralytics)
            - PyTorch
            - OpenCV
            - NumPy
            """)
            
        with tech_col2:
            st.markdown("""
            **Web Application:**
            - Streamlit
            - Plotly
            - Pillow (PIL)
            - Pandas
            """)
    
    with col2:
        # Project Statistics (Mock data for demo)
        st.markdown("""
        <div class="stats-container">
            <h3 style="text-align: center; margin-bottom: 1.5rem;">📊 Project Impact</h3>
            <div style="text-align: center;">
                <div style="font-size: 2.5em; font-weight: bold;">10,000+</div>
                <div style="margin-bottom: 1rem;">Images Analyzed</div>
                
                <div style="font-size: 2.5em; font-weight: bold;">95%</div>
                <div style="margin-bottom: 1rem;">Detection Accuracy</div>
                
                <div style="font-size: 2.5em; font-weight: bold;">3</div>
                <div style="margin-bottom: 1rem;">Waste Categories</div>
                
                <div style="font-size: 2.5em; font-weight: bold;">24/7</div>
                <div>Monitoring Capability</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("## ✨ Key Features")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Smart Detection</h4>
            <p>Advanced YOLOv8 model trained to identify plastic, paper, and organic waste with high accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Visual Analytics</h4>
            <p>Interactive charts and heatmaps provide insights into waste distribution and detection patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🚀 Real-time Processing</h4>
            <p>Fast inference and immediate results with progress tracking and animated feedback.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Waste Categories
    st.markdown("## 🗑️ Waste Categories")
    
    cat_col1, cat_col2, cat_col3 = st.columns(3)
    
    with cat_col1:
        st.markdown("""
        ### 🔵 Plastic Waste
        - Bottles and containers
        - Plastic bags and packaging
        - Disposable utensils
        - Food containers
        
        **Color Code:** Blue bounding boxes
        """)
    
    with cat_col2:
        st.markdown("""
        ### 🟡 Paper Waste
        - Newspapers and magazines
        - Cardboard boxes
        - Paper bags
        - Documents and books
        
        **Color Code:** Yellow bounding boxes
        """)
    
    with cat_col3:
        st.markdown("""
        ### 🟢 Organic Waste
        - Food scraps and leftovers
        - Fruit and vegetable peels
        - Biodegradable materials
        - Compostable items
        
        **Color Code:** Green bounding boxes
        """)
    
    # How It Works
    st.markdown("## 🔄 How It Works")
    
    # Create a flow diagram using Plotly
    fig = go.Figure()
    
    # Add shapes for the flow
    steps = [
        "📤 Upload Image",
        "🔍 AI Analysis", 
        "🎯 Object Detection",
        "📊 Generate Results"
    ]
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
    
    for i, (step, color) in enumerate(zip(steps, colors)):
        fig.add_shape(
            type="rect",
            x0=i*2, y0=0, x1=i*2+1.5, y1=1,
            fillcolor=color,
            line=dict(color=color)
        )
        
        fig.add_annotation(
            x=i*2+0.75, y=0.5,
            text=step,
            showarrow=False,
            font=dict(color="white", size=12, family="Arial Black"),
            align="center"
        )
        
        if i < len(steps) - 1:
            fig.add_annotation(
                x=i*2+1.75, y=0.5,
                text="→",
                showarrow=False,
                font=dict(size=20, color="#333"),
                align="center"
            )
    
    fig.update_layout(
        title="Processing Pipeline",
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.5, 7.5]),
        yaxis=dict(visible=False, range=[-0.5, 1.5]),
        height=200,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Applications
    st.markdown("## 🏙️ Applications")
    
    app_col1, app_col2 = st.columns(2)
    
    with app_col1:
        st.markdown("""
        ### Municipal Use Cases
        - **Street Cleaning Optimization**: Identify high-waste areas for targeted cleaning
        - **Resource Allocation**: Deploy cleaning crews more efficiently
        - **Progress Monitoring**: Track cleanliness improvements over time
        - **Policy Planning**: Data-driven decisions for waste management policies
        """)
    
    with app_col2:
        st.markdown("""
        ### Environmental Applications
        - **Research Studies**: Analyze urban waste patterns and trends
        - **Citizen Engagement**: Enable community reporting of waste issues
        - **Education**: Raise awareness about different waste types
        - **Compliance Monitoring**: Ensure adherence to waste disposal regulations
        """)
    
    # Future Roadmap
    st.markdown("## 🚀 Future Roadmap")
    
    roadmap_items = [
        "🎥 **Real-time Video Processing**: Live monitoring of waste accumulation",
        "🌐 **Multi-language Support**: Expand accessibility globally", 
        "📱 **Mobile Application**: On-the-go waste detection and reporting",
        "🤖 **Advanced AI Models**: Improved accuracy and new waste categories",
        "🗺️ **GIS Integration**: Mapping and geographic analysis capabilities",
        "☁️ **Cloud Deployment**: Scalable infrastructure for large-scale monitoring"
    ]
    
    for item in roadmap_items:
        st.markdown(f"- {item}")
    
    # Contact and Support
    st.markdown("---")
    st.markdown("## 📞 Contact & Support")
    
    contact_col1, contact_col2, contact_col3 = st.columns(3)
    
    with contact_col1:
        st.markdown("""
        **🛠️ Technical Support**
        - GitHub Issues
        - Documentation
        - Community Forum
        """)
    
    with contact_col2:
        st.markdown("""
        **🤝 Partnerships**
        - Municipal Collaborations
        - Research Institutions
        - Environmental Organizations
        """)
    
    with contact_col3:
        st.markdown("""
        **📧 General Inquiries**
        - Project Information
        - Feature Requests
        - Feedback & Suggestions
        """)

if __name__ == "__main__":
    main()
