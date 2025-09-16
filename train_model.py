#!/usr/bin/env python3
"""
YOLOv8 Training Script for Clean City Waste Detection
Trains on Kaggle Garbage Classification v2 dataset
"""

import os
import torch
import yaml
from ultralytics import YOLO
from pathlib import Path
import json
import time
from datetime import datetime

class WasteModelTrainer:
    """Handles YOLO model training for waste detection"""
    
    def __init__(self, dataset_path="dataset.yaml"):
        self.dataset_path = dataset_path
        self.model_save_path = "yolo_model/yolov8_waste_trained.pt"
        self.results_path = "training_results"
        
        # Create results directory
        Path(self.results_path).mkdir(exist_ok=True)
        
    def check_dataset(self):
        """Verify dataset exists and is properly formatted"""
        print("🔍 Checking dataset...")
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset config not found: {self.dataset_path}")
            return False
            
        # Load dataset config
        with open(self.dataset_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required paths
        required_paths = ['train', 'val']
        for path_key in required_paths:
            if path_key not in config:
                print(f"❌ Missing {path_key} path in dataset config")
                return False
                
            # Check if images exist
            img_path = f"dataset/{config[path_key]}"
            if not os.path.exists(img_path):
                print(f"❌ Image path not found: {img_path}")
                return False
                
            # Count images
            img_count = len([f for f in os.listdir(img_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            print(f"✅ Found {img_count} images in {path_key} set")
            
            if img_count == 0:
                print(f"❌ No images found in {path_key} set")
                return False
        
        print("✅ Dataset validation passed!")
        return True
    
    def setup_training_environment(self):
        """Setup training environment and check GPU availability"""
        print("🖥️ Setting up training environment...")
        
        # Check PyTorch installation
        print(f"PyTorch version: {torch.__version__}")
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            device = 'cuda'
        else:
            print("⚠️ No GPU available, using CPU (training will be slower)")
            device = 'cpu'
            
        return device
    
    def train_model(self, epochs=50, batch_size=16, img_size=640):
        """Train the YOLO model"""
        print("🚀 Starting model training...")
        
        # Check dataset first
        if not self.check_dataset():
            return False
            
        # Setup environment
        device = self.setup_training_environment()
        
        try:
            # Load pretrained YOLOv8 model
            print("📦 Loading pretrained YOLOv8 model...")
            
            # Try to use local model first, then download if needed
            import urllib.request
            import ssl
            
            # Load YOLOv8 model - updated ultralytics should work now
            try:
                print("🔄 Initializing YOLOv8n model...")
                model = YOLO('yolov8n.pt')  # This will auto-download if needed
                print("✅ YOLOv8 model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load YOLO model: {e}")
                return False
            
            # Start training
            print(f"🎯 Training parameters:")
            print(f"   - Epochs: {epochs}")
            print(f"   - Batch size: {batch_size}")
            print(f"   - Image size: {img_size}")
            print(f"   - Device: {device}")
            
            start_time = time.time()
            
            # Train the model
            results = model.train(
                data=self.dataset_path,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                patience=10,        # Early stopping patience
                save=True,
                plots=True,
                val=True,
                project=self.results_path,
                name='waste_detection_training',
                exist_ok=True
            )
            
            training_time = time.time() - start_time
            
            # Save the trained model
            print("💾 Saving trained model...")
            model.save(self.model_save_path)
            
            # Validate the model
            print("📊 Running final validation...")
            metrics = model.val()
            
            # Save training summary
            self.save_training_summary(results, metrics, training_time, epochs, batch_size, img_size)
            
            print(f"\n🎉 Training completed successfully!")
            print(f"⏱️ Training time: {training_time/3600:.2f} hours")
            print(f"📁 Model saved to: {self.model_save_path}")
            print(f"📊 mAP50: {metrics.box.map50:.3f}")
            print(f"📊 mAP50-95: {metrics.box.map:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            return False
    
    def save_training_summary(self, results, metrics, training_time, epochs, batch_size, img_size):
        """Save training summary to JSON file"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'training_time_hours': training_time / 3600,
            'parameters': {
                'epochs': epochs,
                'batch_size': batch_size,
                'image_size': img_size,
                'model_type': 'YOLOv8n'
            },
            'metrics': {
                'map50': float(metrics.box.map50),
                'map50_95': float(metrics.box.map),
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr)
            },
            'model_path': self.model_save_path
        }
        
        summary_file = os.path.join(self.results_path, 'training_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Training summary saved to: {summary_file}")

def main():
    """Main training function"""
    print("🗑️ Clean City Waste Detection - Model Training")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists('dataset.yaml'):
        print("❌ Dataset not found!")
        print("Please run: python download_dataset.py")
        print("And choose option 4 to convert your Kaggle dataset")
        return
    
    # Initialize trainer
    trainer = WasteModelTrainer()
    
    # Get training parameters from user
    print("\n⚙️ Training Configuration:")
    
    try:
        epochs = int(input("Enter number of epochs (default 50): ") or "50")
        batch_size = int(input("Enter batch size (default 16): ") or "16")
        img_size = int(input("Enter image size (default 640): ") or "640")
    except ValueError:
        print("Using default parameters...")
        epochs, batch_size, img_size = 50, 16, 640
    
    # Confirm training
    print(f"\n📋 Training Summary:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Image Size: {img_size}")
    print(f"   - Estimated Time: {epochs * 2:.0f}-{epochs * 5:.0f} minutes")
    
    print("\n🚀 Starting training automatically...")
    time.sleep(1)
    
    # Start training
    success = trainer.train_model(epochs, batch_size, img_size)
    
    if success:
        print("\n🎊 Training completed successfully!")
        print("\n📋 Next steps:")
        print("1. Check training results in 'training_results' folder")
        print("2. Your trained model is saved as 'yolo_model/yolov8_waste_trained.pt'")
        print("3. Update Streamlit app to use your trained model")
        print("4. Test the model with new images")
    else:
        print("\n❌ Training failed. Check the error messages above.")

if __name__ == "__main__":
    main()
