"""
Visualization Module for Clean City Waste Detection
Handles image overlays, bounding boxes, and interactive charts
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional
import io
import base64

class WasteVisualizer:
    """
    Handles all visualization tasks for waste detection results
    """
    
    def __init__(self):
        """Initialize the visualizer with color schemes and styling"""
        self.class_colors = {
            'plastic': (255, 0, 0),    # Blue (BGR for OpenCV)
            'paper': (0, 255, 255),    # Yellow
            'organic': (0, 255, 0)     # Green
        }
        
        # RGB colors for Plotly
        self.plotly_colors = {
            'plastic': '#FF6B6B',      # Red
            'paper': '#FFE66D',        # Yellow  
            'organic': '#4ECDC4'       # Teal
        }
        
        self.font_scale = 0.7
        self.thickness = 2
        
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image (BGR format)
            detections: List of detection dictionaries
            
        Returns:
            Image with drawn bounding boxes
        """
        if not detections:
            return image
            
        # Create a copy to avoid modifying original
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            color = detection['color']
            
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, self.thickness)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness
            )
            
            # Draw label background
            cv2.rectangle(
                result_image,
                (x1, y1 - text_height - baseline - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                result_image,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),  # White text
                self.thickness
            )
        
        return result_image
    
    def create_pie_chart(self, class_counts: Dict[str, int]) -> go.Figure:
        """
        Create an interactive pie chart showing waste distribution
        
        Args:
            class_counts: Dictionary with waste class counts
            
        Returns:
            Plotly figure object
        """
        # Filter out zero counts
        filtered_counts = {k: v for k, v in class_counts.items() if v > 0}
        
        if not filtered_counts:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No waste detected",
                x=0.5, y=0.5,
                font=dict(size=20),
                showarrow=False
            )
            fig.update_layout(
                title="Waste Distribution",
                showlegend=False,
                height=400
            )
            return fig
        
        labels = list(filtered_counts.keys())
        values = list(filtered_counts.values())
        colors = [self.plotly_colors[label] for label in labels]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
            textinfo='label+percent+value',
            textfont=dict(size=14),
            hovertemplate='<b>%{label}</b><br>' +
                         'Count: %{value}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(
                text="🗑️ Waste Distribution Analysis",
                x=0.5,
                font=dict(size=20, color='#2E4057')
            ),
            font=dict(family="Arial, sans-serif"),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            ),
            height=500,
            margin=dict(t=80, b=40, l=40, r=120),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_detection_summary(self, detection_results: Dict) -> go.Figure:
        """
        Create a summary bar chart of detection results
        
        Args:
            detection_results: Detection results dictionary
            
        Returns:
            Plotly bar chart figure
        """
        class_counts = detection_results['class_counts']
        
        categories = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = [self.plotly_colors[cat] for cat in categories]
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=counts,
                marker_color=colors,
                text=counts,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(
                text="📊 Detection Summary",
                x=0.5,
                font=dict(size=18, color='#2E4057')
            ),
            xaxis_title="Waste Type",
            yaxis_title="Count",
            font=dict(family="Arial, sans-serif"),
            height=400,
            margin=dict(t=60, b=40, l=40, r=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray')
        )
        
        return fig
    
    def create_confidence_distribution(self, detections: List[Dict]) -> go.Figure:
        """
        Create histogram of detection confidences
        
        Args:
            detections: List of detection results
            
        Returns:
            Plotly histogram figure
        """
        if not detections:
            fig = go.Figure()
            fig.add_annotation(
                text="No detections to analyze",
                x=0.5, y=0.5,
                font=dict(size=16),
                showarrow=False
            )
            fig.update_layout(
                title="Confidence Distribution",
                height=300
            )
            return fig
        
        confidences = [det['confidence'] for det in detections]
        classes = [det['class'] for det in detections]
        
        fig = go.Figure()
        
        for waste_class in ['plastic', 'paper', 'organic']:
            class_confidences = [conf for conf, cls in zip(confidences, classes) if cls == waste_class]
            if class_confidences:
                fig.add_trace(go.Histogram(
                    x=class_confidences,
                    name=waste_class.capitalize(),
                    marker_color=self.plotly_colors[waste_class],
                    opacity=0.7,
                    nbinsx=10
                ))
        
        fig.update_layout(
            title=dict(
                text="🎯 Detection Confidence Distribution",
                x=0.5,
                font=dict(size=16, color='#2E4057')
            ),
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            barmode='overlay',
            height=350,
            margin=dict(t=60, b=40, l=40, r=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_detection_heatmap(self, image_shape: Tuple[int, int], detections: List[Dict]) -> go.Figure:
        """
        Create a heatmap showing detection density across the image
        
        Args:
            image_shape: (height, width) of the original image
            detections: List of detection results
            
        Returns:
            Plotly heatmap figure
        """
        height, width = image_shape
        
        # Create a grid for the heatmap
        grid_size = 20
        heatmap = np.zeros((grid_size, grid_size))
        
        if detections:
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Map to grid coordinates
                grid_x = min(int((center_x / width) * grid_size), grid_size - 1)
                grid_y = min(int((center_y / height) * grid_size), grid_size - 1)
                
                heatmap[grid_y, grid_x] += 1
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap,
            colorscale='Viridis',
            showscale=True,
            hoverongaps=False,
            hovertemplate='Grid X: %{x}<br>Grid Y: %{y}<br>Detections: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text="🔥 Detection Density Heatmap",
                x=0.5,
                font=dict(size=16, color='#2E4057')
            ),
            height=400,
            margin=dict(t=60, b=40, l=40, r=40),
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Image Width (Grid)"),
            yaxis=dict(title="Image Height (Grid)")
        )
        
        return fig
    
    def create_animated_progress_bar(self, progress: float, label: str = "Processing") -> str:
        """
        Create an animated progress bar using HTML/CSS
        
        Args:
            progress: Progress value between 0 and 1
            label: Label for the progress bar
            
        Returns:
            HTML string for the progress bar
        """
        progress_percent = int(progress * 100)
        
        html = f"""
        <div style="margin: 20px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: bold; color: #2E4057;">{label}</span>
                <span style="color: #666;">{progress_percent}%</span>
            </div>
            <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; overflow: hidden;">
                <div style="
                    background: linear-gradient(90deg, #4ECDC4, #44A08D);
                    height: 100%;
                    width: {progress_percent}%;
                    border-radius: 10px;
                    transition: width 0.3s ease;
                    animation: pulse 2s infinite;
                "></div>
            </div>
        </div>
        <style>
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
            100% {{ opacity: 1; }}
        }}
        </style>
        """
        return html
    
    def convert_cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """
        Convert OpenCV image (BGR) to PIL Image (RGB)
        
        Args:
            cv2_image: OpenCV image in BGR format
            
        Returns:
            PIL Image in RGB format
        """
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    
    def create_detection_overlay_gif(self, images: List[np.ndarray], detections_list: List[List[Dict]]) -> bytes:
        """
        Create an animated GIF showing detection results over multiple frames
        
        Args:
            images: List of images
            detections_list: List of detection results for each image
            
        Returns:
            GIF as bytes
        """
        pil_images = []
        
        for image, detections in zip(images, detections_list):
            # Draw detections on image
            annotated_image = self.draw_detections(image, detections)
            # Convert to PIL
            pil_image = self.convert_cv2_to_pil(annotated_image)
            pil_images.append(pil_image)
        
        # Save as GIF
        output = io.BytesIO()
        if pil_images:
            pil_images[0].save(
                output,
                format='GIF',
                save_all=True,
                append_images=pil_images[1:],
                duration=500,  # 500ms per frame
                loop=0
            )
        
        return output.getvalue()


def create_detection_stats_card(detection_results: Dict) -> str:
    """
    Create an HTML card displaying detection statistics
    
    Args:
        detection_results: Detection results dictionary
        
    Returns:
        HTML string for the stats card
    """
    total = detection_results['total_objects']
    plastic = detection_results['class_counts']['plastic']
    paper = detection_results['class_counts']['paper']
    organic = detection_results['class_counts']['organic']
    
    html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    ">
        <h3 style="margin: 0 0 15px 0; text-align: center;">🎯 Detection Summary</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="text-align: center;">
                <div style="font-size: 2em; font-weight: bold;">{total}</div>
                <div style="opacity: 0.9;">Total Objects</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2em; font-weight: bold; color: #FF6B6B;">{plastic}</div>
                <div style="opacity: 0.9;">🥤 Plastic</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2em; font-weight: bold; color: #FFE66D;">{paper}</div>
                <div style="opacity: 0.9;">📄 Paper</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2em; font-weight: bold; color: #4ECDC4;">{organic}</div>
                <div style="opacity: 0.9;">🍎 Organic</div>
            </div>
        </div>
    </div>
    """
    return html
