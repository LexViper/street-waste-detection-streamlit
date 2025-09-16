#!/usr/bin/env python3
"""
Automatic Dataset Downloader for Clean City Waste Detection
Downloads and prepares the TACO dataset for training
"""

import os
import requests
import zipfile
import json
from pathlib import Path
import shutil

def download_file(url, filename):
    """Download file with progress bar"""
    print(f"📥 Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r📊 Progress: {percent:.1f}%", end='', flush=True)
    print("\n✅ Download complete!")

def setup_directories():
    """Create necessary directories"""
    dirs = [
        'dataset/images/train',
        'dataset/images/val', 
        'dataset/labels/train',
        'dataset/labels/val',
        'temp'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("📁 Created directory structure")

def download_taco_dataset():
    """Download TACO dataset"""
    print("🗑️ Starting TACO Dataset Download...")
    
    # TACO dataset URLs
    urls = {
        'annotations': 'http://tacodataset.org/files/annotations.json',
        'images_batch_1': 'http://tacodataset.org/files/batch_1.zip',
        'images_batch_2': 'http://tacodataset.org/files/batch_2.zip',
        'images_batch_3': 'http://tacodataset.org/files/batch_3.zip'
    }
    
    setup_directories()
    
    # Download annotations
    download_file(urls['annotations'], 'temp/annotations.json')
    
    # Download image batches
    for batch_name, url in urls.items():
        if batch_name.startswith('images_batch'):
            download_file(url, f'temp/{batch_name}.zip')
            
            # Extract images
            print(f"📦 Extracting {batch_name}...")
            with zipfile.ZipFile(f'temp/{batch_name}.zip', 'r') as zip_ref:
                zip_ref.extractall('temp/images')
            
            # Clean up zip file
            os.remove(f'temp/{batch_name}.zip')
    
    print("✅ TACO dataset downloaded successfully!")

def download_roboflow_sample():
    """Download a sample Roboflow waste detection dataset"""
    print("🤖 Downloading Roboflow sample dataset...")
    
    # This is a sample dataset - replace with actual Roboflow dataset URL
    sample_url = "https://app.roboflow.com/ds/your-dataset-id"
    
    print("""
    📋 To download from Roboflow:
    1. Visit: https://universe.roboflow.com/
    2. Search: 'waste detection'
    3. Choose a dataset (e.g., 'Waste Detection Computer Vision Project')
    4. Click 'Download Dataset'
    5. Select 'YOLOv8' format
    6. Copy download link and update this script
    """)

def create_dataset_yaml():
    """Create dataset.yaml file for YOLO training"""
    yaml_content = """# Clean City Waste Detection Dataset
path: ./dataset  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
names:
  0: plastic
  1: paper
  2: organic

# Number of classes
nc: 3
"""
    
    with open('dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("📄 Created dataset.yaml configuration file")

def convert_coco_to_yolo():
    """Convert COCO annotations to YOLO format"""
    print("🔄 Converting annotations to YOLO format...")
    
    # Load COCO annotations
    with open('temp/annotations.json', 'r') as f:
        coco_data = json.load(f)
    
    # Map TACO categories to our 3 classes
    category_mapping = {
        # Plastic items
        'Plastic bottle': 0,
        'Plastic bag & wrapper': 0,
        'Plastic container': 0,
        'Plastic cup': 0,
        'Plastic lid': 0,
        'Plastic utensils': 0,
        
        # Paper items
        'Paper': 1,
        'Cardboard': 1,
        'Paper bag': 1,
        'Magazine paper': 1,
        'Newspaper': 1,
        
        # Organic items
        'Food waste': 2,
        'Banana peel': 2,
        'Apple core': 2,
    }
    
    # Create category lookup
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Process annotations
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Convert images and annotations
    train_count = 0
    val_count = 0
    
    for img in coco_data['images']:
        img_id = img['id']
        img_name = img['file_name']
        img_width = img['width']
        img_height = img['height']
        
        # Determine split (80% train, 20% val)
        is_train = (img_id % 5) != 0
        split = 'train' if is_train else 'val'
        
        # Copy image file
        src_path = f"temp/images/{img_name}"
        dst_path = f"dataset/images/{split}/{img_name}"
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            
            # Convert annotations
            yolo_annotations = []
            if img_id in image_annotations:
                for ann in image_annotations[img_id]:
                    cat_name = categories.get(ann['category_id'], 'Unknown')
                    class_id = category_mapping.get(cat_name)
                    
                    if class_id is not None:
                        # Convert COCO bbox to YOLO format
                        x, y, w, h = ann['bbox']
                        x_center = (x + w/2) / img_width
                        y_center = (y + h/2) / img_height
                        width = w / img_width
                        height = h / img_height
                        
                        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save YOLO annotation file
            if yolo_annotations:
                ann_path = f"dataset/labels/{split}/{os.path.splitext(img_name)[0]}.txt"
                with open(ann_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
            
            if is_train:
                train_count += 1
            else:
                val_count += 1
    
    print(f"✅ Converted {train_count} training images and {val_count} validation images")

def cleanup_temp():
    """Clean up temporary files"""
    if os.path.exists('temp'):
        shutil.rmtree('temp')
    print("🧹 Cleaned up temporary files")

def convert_kaggle_dataset():
    """Convert Kaggle Garbage Classification v2 dataset to YOLO format"""
    print("🔄 Converting Kaggle dataset to YOLO format...")
    
    # Category mapping for Kaggle dataset
    category_mapping = {
        'plastic': 0,
        'paper': 1, 
        'cardboard': 1,  # Map cardboard to paper
        'organic': 2,
        'metal': 0,      # Map metal to plastic
        'glass': 0,      # Map glass to plastic
        'trash': 0       # General trash to plastic
    }
    
    setup_directories()
    
    # Look for Kaggle dataset folder
    kaggle_folders = ['garbage-dataset', 'garbage-classification-v2', 'archive', 'dataset']
    dataset_root = None
    
    for folder in kaggle_folders:
        if os.path.exists(folder):
            dataset_root = folder
            break
    
    if not dataset_root:
        print("❌ Kaggle dataset not found. Please ensure it's downloaded and extracted.")
        return False
    
    print(f"📂 Found Kaggle dataset at: {dataset_root}")
    
    # Process each category
    train_count = 0
    val_count = 0
    
    for category in os.listdir(dataset_root):
        category_path = os.path.join(dataset_root, category)
        
        if not os.path.isdir(category_path):
            continue
            
        category_lower = category.lower()
        if category_lower not in category_mapping:
            print(f"⚠️ Skipping unknown category: {category}")
            continue
            
        class_id = category_mapping[category_lower]
        print(f"📋 Processing {category} → class {class_id}")
        
        # Get all images in category
        image_files = [f for f in os.listdir(category_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Shuffle for random train/val split
        import random
        random.shuffle(image_files)
        
        # 80% train, 20% validation
        split_idx = int(0.8 * len(image_files))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Process training images
        for img_file in train_files:
            src_path = os.path.join(category_path, img_file)
            dst_path = f"dataset/images/train/{category}_{img_file}"
            
            try:
                shutil.copy2(src_path, dst_path)
                
                # Create YOLO annotation (full image as one object)
                ann_file = f"dataset/labels/train/{category}_{os.path.splitext(img_file)[0]}.txt"
                with open(ann_file, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")
                
                train_count += 1
            except Exception as e:
                print(f"⚠️ Error processing {img_file}: {e}")
        
        # Process validation images
        for img_file in val_files:
            src_path = os.path.join(category_path, img_file)
            dst_path = f"dataset/images/val/{category}_{img_file}"
            
            try:
                shutil.copy2(src_path, dst_path)
                
                # Create YOLO annotation
                ann_file = f"dataset/labels/val/{category}_{os.path.splitext(img_file)[0]}.txt"
                with open(ann_file, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")
                
                val_count += 1
            except Exception as e:
                print(f"⚠️ Error processing {img_file}: {e}")
    
    print(f"✅ Converted {train_count} training images and {val_count} validation images")
    return True

def main():
    """Main function to download and prepare dataset"""
    print("🌍 Clean City Dataset Downloader")
    print("=" * 50)
    
    choice = input("""
Choose dataset option:
1. TACO Dataset (Recommended - 1.5GB)
2. Roboflow Sample (Instructions only)
3. Create sample dataset for testing
4. Convert Kaggle dataset (if already downloaded)

Enter choice (1-4): """).strip()
    
    if choice == '1':
        try:
            download_taco_dataset()
            convert_coco_to_yolo()
            create_dataset_yaml()
            cleanup_temp()
            
            print("\n🎉 Dataset setup complete!")
            print("📊 Dataset statistics:")
            print(f"   - Training images: {len(os.listdir('dataset/images/train'))}")
            print(f"   - Validation images: {len(os.listdir('dataset/images/val'))}")
            print("\n🚀 Ready to train! Run:")
            print("   python train_model.py")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Try option 3 for a sample dataset")
    
    elif choice == '2':
        download_roboflow_sample()
    
    elif choice == '3':
        create_sample_dataset()
        
    elif choice == '4':
        try:
            if convert_kaggle_dataset():
                create_dataset_yaml()
                print("\n🎉 Kaggle dataset conversion complete!")
                print("📊 Dataset statistics:")
                train_count = len([f for f in os.listdir('dataset/images/train') if f.endswith(('.jpg', '.jpeg', '.png'))])
                val_count = len([f for f in os.listdir('dataset/images/val') if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f"   - Training images: {train_count}")
                print(f"   - Validation images: {val_count}")
                print("\n🚀 Ready to train! Run:")
                print("   python train_model.py")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    else:
        print("❌ Invalid choice")

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    print("📦 Creating sample dataset for testing...")
    
    setup_directories()
    create_dataset_yaml()
    
    # Create some dummy annotation files
    sample_annotations = [
        "0 0.5 0.5 0.2 0.3",  # plastic bottle
        "1 0.3 0.7 0.15 0.2", # paper
        "2 0.8 0.4 0.1 0.15"  # organic
    ]
    
    # Create sample label files
    for i in range(5):
        # Training samples
        with open(f'dataset/labels/train/sample_{i}.txt', 'w') as f:
            f.write(sample_annotations[i % 3])
        
        # Validation samples  
        with open(f'dataset/labels/val/sample_{i}.txt', 'w') as f:
            f.write(sample_annotations[i % 3])
    
    print("✅ Sample dataset created!")
    print("📝 Add your own images to dataset/images/train and dataset/images/val")
    print("📝 Add corresponding .txt annotation files to dataset/labels/")

if __name__ == "__main__":
    main()
