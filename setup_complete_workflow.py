#!/usr/bin/env python3
"""
Complete Workflow Setup for Clean City Waste Detection
Automates the entire process from dataset to trained model
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def print_step(step_num, title, description):
    """Print formatted step information"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*60}")
    print(description)
    print()

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'streamlit', 'ultralytics', 'opencv-python', 'pillow', 
        'numpy', 'pandas', 'matplotlib', 'plotly', 'torch'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All requirements are satisfied")
    return True

def setup_directories():
    """Create all necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        'dataset/images/train',
        'dataset/images/val',
        'dataset/labels/train',
        'dataset/labels/val',
        'yolo_model',
        'training_results',
        'evaluation_results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def main():
    """Main workflow setup"""
    print("🌍 Clean City Waste Detection - Complete Workflow Setup")
    print("This script will guide you through the entire process")
    
    # Step 1: Check Requirements
    print_step(1, "REQUIREMENTS CHECK", 
               "Checking if all Python packages are installed")
    
    if not check_requirements():
        print("❌ Please install requirements first: pip install -r requirements.txt")
        return
    
    # Step 2: Setup Directories
    print_step(2, "DIRECTORY SETUP", 
               "Creating necessary directories for dataset and models")
    
    setup_directories()
    
    # Step 3: Dataset Instructions
    print_step(3, "DATASET PREPARATION", 
               "Instructions for downloading and preparing the Kaggle dataset")
    
    print("""
📋 DATASET SETUP INSTRUCTIONS:

1. Download Kaggle Dataset:
   - Go to: https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
   - Click 'Download' (you may need to create a Kaggle account)
   - Extract the downloaded ZIP file to this project folder
   
2. Convert Dataset:
   - Run: python3 download_dataset.py
   - Choose option 4: "Convert Kaggle dataset"
   - Wait for conversion to complete
   
3. Verify Dataset:
   - Check that 'dataset' folder contains images and labels
   - Training and validation splits should be created automatically
    """)
    
    dataset_ready = input("\n✅ Have you completed the dataset setup? (y/n): ").lower().strip()
    
    if dataset_ready != 'y':
        print("⏸️ Please complete dataset setup first, then run this script again")
        return
    
    # Step 4: Training
    print_step(4, "MODEL TRAINING", 
               "Training the YOLOv8 model on your dataset")
    
    if os.path.exists('dataset.yaml'):
        train_now = input("🚀 Start training now? This may take 30-60 minutes (y/n): ").lower().strip()
        
        if train_now == 'y':
            print("🎯 Starting model training...")
            print("⏱️ This will take some time. You can monitor progress in the terminal.")
            
            # Run training
            success = run_command("python3 train_model.py", "Model training")
            
            if success:
                print("🎉 Training completed successfully!")
            else:
                print("❌ Training failed. Check the error messages above.")
        else:
            print("⏸️ You can start training later with: python3 train_model.py")
    else:
        print("❌ Dataset configuration not found. Please complete dataset setup first.")
    
    # Step 5: Evaluation
    print_step(5, "MODEL EVALUATION", 
               "Evaluating the trained model performance")
    
    if os.path.exists('yolo_model/yolov8_waste_trained.pt'):
        eval_now = input("📊 Run model evaluation? (y/n): ").lower().strip()
        
        if eval_now == 'y':
            success = run_command("python3 evaluate_model.py", "Model evaluation")
            
            if success:
                print("📈 Evaluation completed! Check evaluation_results/ folder")
    else:
        print("⏸️ No trained model found. Complete training first.")
    
    # Step 6: Launch Application
    print_step(6, "LAUNCH APPLICATION", 
               "Starting the Streamlit web application")
    
    launch_app = input("🚀 Launch the Clean City app now? (y/n): ").lower().strip()
    
    if launch_app == 'y':
        print("🌐 Starting Streamlit application...")
        print("📱 The app will open in your browser automatically")
        print("🔗 If not, navigate to: http://localhost:8501")
        print("\n⚠️ Press Ctrl+C to stop the application")
        
        try:
            subprocess.run(["streamlit", "run", "streamlit_app.py"], check=True)
        except KeyboardInterrupt:
            print("\n👋 Application stopped")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start application: {e}")
    
    # Final Summary
    print("\n" + "="*60)
    print("🎊 SETUP COMPLETE!")
    print("="*60)
    
    print("""
📋 WHAT'S BEEN SET UP:

✅ Project Structure: All directories created
✅ Dataset Tools: Kaggle dataset converter ready
✅ Training Pipeline: YOLOv8 training script ready
✅ Evaluation Tools: Model performance analysis ready
✅ Web Application: Multi-page Streamlit app ready

🚀 NEXT STEPS:

1. Complete dataset setup if not done
2. Train your model: python3 train_model.py
3. Evaluate performance: python3 evaluate_model.py
4. Launch the app: streamlit run streamlit_app.py

📁 KEY FILES:
- streamlit_app.py: Main application
- train_model.py: Training script
- evaluate_model.py: Evaluation script
- download_dataset.py: Dataset management
- pages/: Additional app pages (Training, Dataset, About, Demo)

🌍 Your Clean City Waste Detection system is ready!
    """)

if __name__ == "__main__":
    main()
