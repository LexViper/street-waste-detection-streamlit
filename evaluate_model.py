#!/usr/bin/env python3
"""
Model Evaluation Script for Clean City Waste Detection
Evaluates trained models and generates detailed performance reports
"""

import os
import json
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ModelEvaluator:
    """Comprehensive model evaluation for waste detection"""
    
    def __init__(self, model_path="yolo_model/yolov8_waste_trained.pt", dataset_path="dataset.yaml"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.results_dir = "evaluation_results"
        self.class_names = {0: 'plastic', 1: 'paper', 2: 'organic'}
        
        # Create results directory
        Path(self.results_dir).mkdir(exist_ok=True)
        
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"📦 Loading model from {self.model_path}")
        self.model = YOLO(self.model_path)
        return True
    
    def evaluate_on_validation_set(self):
        """Run evaluation on validation dataset"""
        print("🔍 Running validation evaluation...")
        
        # Run validation
        results = self.model.val(data=self.dataset_path, save_json=True, save_hybrid=True)
        
        # Extract metrics
        metrics = {
            'map50': float(results.box.map50),
            'map50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'fitness': float(results.fitness)
        }
        
        # Per-class metrics
        if hasattr(results.box, 'maps'):
            class_metrics = {}
            for i, class_name in self.class_names.items():
                if i < len(results.box.maps):
                    class_metrics[class_name] = {
                        'map50': float(results.box.maps[i]),
                        'precision': float(results.box.p[i]) if hasattr(results.box, 'p') else 0,
                        'recall': float(results.box.r[i]) if hasattr(results.box, 'r') else 0
                    }
            metrics['per_class'] = class_metrics
        
        return metrics
    
    def test_on_sample_images(self, sample_dir="dataset/images/val", num_samples=20):
        """Test model on sample images and analyze results"""
        print(f"🖼️ Testing on {num_samples} sample images...")
        
        if not os.path.exists(sample_dir):
            print(f"❌ Sample directory not found: {sample_dir}")
            return {}
        
        # Get sample images
        image_files = [f for f in os.listdir(sample_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(image_files) == 0:
            print("❌ No images found in sample directory")
            return {}
        
        # Randomly select samples
        np.random.seed(42)
        selected_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        results_data = []
        confidence_scores = []
        detection_counts = {'plastic': 0, 'paper': 0, 'organic': 0}
        
        for img_file in selected_files:
            img_path = os.path.join(sample_dir, img_file)
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Run inference
            results = self.model(image, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for conf, class_id in zip(confidences, class_ids):
                        confidence_scores.append(float(conf))
                        if class_id in self.class_names:
                            detection_counts[self.class_names[class_id]] += 1
                    
                    results_data.append({
                        'image': img_file,
                        'detections': len(boxes),
                        'avg_confidence': float(np.mean(confidences)),
                        'classes_detected': [self.class_names.get(int(c), 'unknown') for c in class_ids]
                    })
                else:
                    results_data.append({
                        'image': img_file,
                        'detections': 0,
                        'avg_confidence': 0,
                        'classes_detected': []
                    })
        
        return {
            'sample_results': results_data,
            'confidence_distribution': confidence_scores,
            'detection_counts': detection_counts,
            'total_samples': len(selected_files)
        }
    
    def create_evaluation_report(self, metrics, sample_results):
        """Create comprehensive evaluation report"""
        print("📊 Generating evaluation report...")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_path': self.model_path,
            'overall_metrics': metrics,
            'sample_analysis': sample_results,
            'summary': {
                'model_performance': 'Good' if metrics.get('map50', 0) > 0.7 else 'Needs Improvement',
                'recommended_actions': []
            }
        }
        
        # Add recommendations based on metrics
        if metrics.get('map50', 0) < 0.5:
            report['summary']['recommended_actions'].append("Consider training for more epochs")
        if metrics.get('precision', 0) < 0.7:
            report['summary']['recommended_actions'].append("Review false positive detections")
        if metrics.get('recall', 0) < 0.7:
            report['summary']['recommended_actions'].append("Add more training data for missed objects")
        
        # Save report
        report_file = os.path.join(self.results_dir, 'evaluation_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Report saved to: {report_file}")
        return report
    
    def create_visualizations(self, metrics, sample_results):
        """Create evaluation visualizations"""
        print("📈 Creating visualizations...")
        
        # 1. Overall metrics bar chart
        fig_metrics = go.Figure(data=[
            go.Bar(
                x=['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95'],
                y=[
                    metrics.get('precision', 0),
                    metrics.get('recall', 0),
                    metrics.get('map50', 0),
                    metrics.get('map50_95', 0)
                ],
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                text=[f"{v:.3f}" for v in [
                    metrics.get('precision', 0),
                    metrics.get('recall', 0),
                    metrics.get('map50', 0),
                    metrics.get('map50_95', 0)
                ]],
                textposition='auto'
            )
        ])
        
        fig_metrics.update_layout(
            title='Model Performance Metrics',
            yaxis_title='Score',
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        fig_metrics.write_html(os.path.join(self.results_dir, 'metrics_chart.html'))
        
        # 2. Per-class performance (if available)
        if 'per_class' in metrics:
            classes = list(metrics['per_class'].keys())
            map50_scores = [metrics['per_class'][c]['map50'] for c in classes]
            
            fig_classes = go.Figure(data=[
                go.Bar(x=classes, y=map50_scores, marker_color=['#FF6B6B', '#FFE66D', '#4ECDC4'])
            ])
            
            fig_classes.update_layout(
                title='Per-Class mAP@0.5 Performance',
                yaxis_title='mAP@0.5',
                height=400
            )
            
            fig_classes.write_html(os.path.join(self.results_dir, 'per_class_performance.html'))
        
        # 3. Confidence distribution
        if sample_results.get('confidence_distribution'):
            confidences = sample_results['confidence_distribution']
            
            fig_conf = go.Figure(data=[
                go.Histogram(x=confidences, nbinsx=20, marker_color='#45B7D1')
            ])
            
            fig_conf.update_layout(
                title='Detection Confidence Distribution',
                xaxis_title='Confidence Score',
                yaxis_title='Frequency',
                height=400
            )
            
            fig_conf.write_html(os.path.join(self.results_dir, 'confidence_distribution.html'))
        
        # 4. Detection counts pie chart
        if sample_results.get('detection_counts'):
            counts = sample_results['detection_counts']
            
            fig_pie = go.Figure(data=[
                go.Pie(
                    labels=list(counts.keys()),
                    values=list(counts.values()),
                    marker_colors=['#FF6B6B', '#FFE66D', '#4ECDC4']
                )
            ])
            
            fig_pie.update_layout(
                title='Detection Distribution in Sample Images',
                height=400
            )
            
            fig_pie.write_html(os.path.join(self.results_dir, 'detection_distribution.html'))
        
        print("✅ Visualizations saved to evaluation_results/")
    
    def run_full_evaluation(self):
        """Run complete model evaluation"""
        print("🎯 Starting comprehensive model evaluation...")
        
        try:
            # Load model
            self.load_model()
            
            # Run validation evaluation
            metrics = self.evaluate_on_validation_set()
            print(f"✅ Validation mAP@0.5: {metrics['map50']:.3f}")
            
            # Test on sample images
            sample_results = self.test_on_sample_images()
            print(f"✅ Tested on {sample_results.get('total_samples', 0)} sample images")
            
            # Create report
            report = self.create_evaluation_report(metrics, sample_results)
            
            # Create visualizations
            self.create_visualizations(metrics, sample_results)
            
            print("\n🎉 Evaluation completed successfully!")
            print(f"📊 Overall Performance: {report['summary']['model_performance']}")
            print(f"📁 Results saved in: {self.results_dir}/")
            
            return report
            
        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            return None

def main():
    """Main evaluation function"""
    print("🔍 Clean City Model Evaluation")
    print("=" * 50)
    
    # Check if trained model exists
    model_path = "yolo_model/yolov8_waste_trained.pt"
    if not os.path.exists(model_path):
        print("❌ Trained model not found!")
        print("Please train a model first using: python train_model.py")
        return
    
    # Check if dataset exists
    if not os.path.exists("dataset.yaml"):
        print("❌ Dataset configuration not found!")
        print("Please prepare dataset first using: python download_dataset.py")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path)
    
    # Run evaluation
    report = evaluator.run_full_evaluation()
    
    if report:
        print("\n📋 Evaluation Summary:")
        print(f"   - mAP@0.5: {report['overall_metrics']['map50']:.3f}")
        print(f"   - Precision: {report['overall_metrics']['precision']:.3f}")
        print(f"   - Recall: {report['overall_metrics']['recall']:.3f}")
        print(f"   - Performance: {report['summary']['model_performance']}")
        
        if report['summary']['recommended_actions']:
            print("\n💡 Recommendations:")
            for action in report['summary']['recommended_actions']:
                print(f"   - {action}")

if __name__ == "__main__":
    main()
