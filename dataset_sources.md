# 🗂️ Waste Detection Datasets

## 📥 Ready-to-Use Datasets

### 1. **TACO Dataset (Trash Annotations in Context)**
- **Size**: 1,500+ images with 4,784 annotations
- **Categories**: 60 waste categories including plastic, paper, organic
- **Format**: COCO format (easily convertible to YOLO)
- **Download**: https://github.com/pedropro/TACO
- **License**: Creative Commons

```bash
# Download TACO dataset
git clone https://github.com/pedropro/TACO.git
cd TACO
python download.py
```

### 2. **TrashNet Dataset**
- **Size**: 2,527 images
- **Categories**: 6 classes (glass, paper, cardboard, plastic, metal, trash)
- **Format**: Classification dataset (can be adapted for detection)
- **Download**: https://github.com/garythung/trashnet
- **License**: MIT

### 3. **Waste Classification Dataset (Kaggle)**
- **Size**: 25,000+ images
- **Categories**: Organic, Recyclable
- **Download**: https://www.kaggle.com/datasets/techsash/waste-classification-data
- **Format**: Classification (needs bounding box annotation)

### 4. **OpenImages Waste Subset**
- **Size**: 1,000+ annotated waste images
- **Categories**: Multiple waste types with bounding boxes
- **Download**: Use OpenImages downloader with waste-related classes
- **Format**: YOLO compatible

## 🛠️ Dataset Creation Tools

### Roboflow (Recommended)
- **URL**: https://roboflow.com/
- **Features**: 
  - Auto-annotation tools
  - Data augmentation
  - Format conversion (COCO → YOLO)
  - Free tier available
- **Waste datasets**: Search "waste detection" in public datasets

### Label Studio
- **URL**: https://labelstud.io/
- **Features**: Open-source annotation tool
- **Good for**: Creating custom annotations

## 📋 Quick Setup Instructions

### Option 1: Use TACO Dataset (Recommended)
```bash
# 1. Download TACO
git clone https://github.com/pedropro/TACO.git
cd TACO
python download.py

# 2. Convert to YOLO format
python scripts/taco_to_yolo.py

# 3. Copy to your project
cp -r yolo_dataset/* /path/to/clean-city-streamlit/yolo_model/
```

### Option 2: Roboflow Public Dataset
1. Visit: https://universe.roboflow.com/
2. Search: "waste detection" or "trash detection"
3. Choose a dataset (e.g., "Waste Detection" by various authors)
4. Download in YOLOv8 format
5. Extract to `yolo_model/` folder

### Option 3: Create Your Own
1. Collect 500-1000 waste images
2. Use Roboflow or Label Studio for annotation
3. Export in YOLOv8 format
4. Train custom model

## 🎯 Recommended Datasets by Use Case

### For Quick Testing
- **TACO Dataset**: Best balance of size and quality
- **Roboflow Public**: Pre-processed and ready to use

### For Production
- **Custom Dataset**: Collect images from your target environment
- **TACO + Custom**: Combine TACO with your specific images

### For Research
- **Multiple Datasets**: Combine TACO, TrashNet, and OpenImages
- **Cross-validation**: Test on different dataset splits

## 📊 Dataset Statistics Comparison

| Dataset | Images | Annotations | Categories | Format | License |
|---------|--------|-------------|------------|--------|---------|
| TACO | 1,500+ | 4,784 | 60 | COCO | CC |
| TrashNet | 2,527 | 2,527 | 6 | Classification | MIT |
| Roboflow | Varies | Varies | 3-10 | YOLO | Varies |
| Custom | Your choice | Your choice | 3+ | YOLO | Your choice |

## 🚀 Training Your Model

Once you have a dataset:

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # pretrained model

# Train the model
results = model.train(
    data='path/to/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)

# Save the trained model
model.save('yolo_model/yolov8_waste.pt')
```

## 💡 Pro Tips

1. **Start Small**: Begin with TACO dataset for proof of concept
2. **Data Quality > Quantity**: 500 good images > 2000 poor images  
3. **Augmentation**: Use Roboflow's augmentation features
4. **Validation**: Keep 20% of data for testing
5. **Iterative**: Start with 3 classes, expand gradually
