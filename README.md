
# 🌍 Clean City Waste Detection

A modern AI-powered web application for detecting and analyzing waste in urban environments using YOLOv8 and Streamlit.

![Clean City Banner](https://img.shields.io/badge/Clean%20City-Waste%20Detection-blue?style=for-the-badge&logo=recycle)
![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=flat-square&logo=streamlit)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=flat-square)

---

## 🎯 Project Overview

**Clean City** is an intelligent waste detection system that helps municipalities, environmental organizations, and citizens identify and analyze waste in urban environments. The application uses state-of-the-art **YOLOv8** object detection to classify waste into three categories:

-   🔵 **Plastic Waste** (bottles, containers, bags)
-   🟡 **Paper Waste** (newspapers, cardboard, documents)
-   🟢 **Organic Waste** (food scraps, biodegradable materials)

This project provides a complete workflow—from dataset preparation and model training to evaluation and deployment via a user-friendly Streamlit web interface.

---

## ✨ Features

-   **AI-Powered Detection**: Leverages the latest YOLOv8 model for high-accuracy waste classification.
-   **Full Workflow**: Includes scripts for automated dataset setup, model training, and performance evaluation.
-   **Advanced Analytics**: Displays interactive charts, including a pie chart showing the percentage breakdown of detected waste types.
-   **Modern UI/UX**: A responsive Streamlit web app that works on desktop and mobile devices.
-   **Modular Design**: Clean, well-documented code with a clear project structure for easy development and extension.

---

## 📸 Screenshots & Examples

Here are some examples of the Clean City application in action, showcasing the main interface, interactive features, and model training results.

### Main Application Interface
The main page allows users to upload images or videos for detection and analysis.
![Main Application Interface](screenshots/main_app_interface.jpg)

### Interactive Features Demo
This section highlights the app's advanced features, including confidence tuning and chart visualizations.
![Interactive Features Demo](screenshots/interactive_features_demo.jpg)

### Model Training Results
The training dashboard provides a clear overview of model performance with key metrics and visualizations.
![Model Training Results](screenshots/training_results.jpg)

---

## 🚀 Quick Start & Complete Workflow Guide

This guide provides a step-by-step process to get the entire system up and running, from preparing the dataset to launching the final application.

### Prerequisites

-   Python 3.8 or higher
-   `pip` package manager
-   4GB+ RAM recommended
-   GPU support is optional but highly recommended for faster training

### Step 1: Clone & Install Dependencies

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/LexViper/clean-city-streamlit.git](https://github.com/LexViper/clean-city-streamlit.git)
    cd clean-city-streamlit
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv clean_city_env
    # On Windows
    .\clean_city_env\Scripts\activate
    # On macOS/Linux
    source clean_city_env/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Dataset Setup & Conversion

Since this workflow is designed to work with a Kaggle dataset, you'll use the provided script to convert it into the required YOLO format.

1.  **Download the Kaggle dataset** and extract the ZIP file to the `clean-city-streamlit` project folder.
2.  **Run the dataset conversion script:**
    ```bash
    python3 download_dataset.py
    ```
    -   Select **Option 4: "Convert Kaggle dataset"** when prompted.
    -   This will create the `dataset/` folder with `images/` and `labels/` subdirectories, ready for training.

### Step 3: Train Your Model

The `train_model.py` script automates the training process using YOLOv8.

```bash
python3 train_model.py
````

  - The script will prompt for training options like epochs, batch size, and image size.
  - **Expected Result**: A trained model file named `yolov8_waste_trained.pt` will be saved in the `yolo_model/` directory.

### Step 4: Evaluate Model Performance

After training, use the evaluation script to check your model's performance.

```bash
python3 evaluate_model.py
```

  - **Expected Result**: This script will generate performance metrics (mAP, precision, recall), confusion matrices, and other visualizations in the `evaluation_results/` folder.

### Step 5: Launch the Application

With the model trained and evaluated, you can now launch the Streamlit web application.

```bash
streamlit run streamlit_app.py
```

  - The app will automatically open in your browser at `http://localhost:8501`.
  - The application will automatically detect and use the newly trained model for all detections.

-----

## 📁 Project Structure

This is a comprehensive overview of the project's folder and file structure.

```
clean-city-streamlit/
│
├── 🎮 streamlit_app.py              # Main Streamlit application
├── 🎯 train_model.py                # YOLOv8 training script
├── 📊 evaluate_model.py             # Model evaluation and metrics
├── 📥 download_dataset.py           # Dataset management and conversion
├── 📄 requirements.txt              # Python dependencies
├── 📚 README.md                     # This file
├── 🗂️ dataset_sources.md           # Dataset options and links
│
├── helpers/                         # Core functionality
│   ├── detection.py                # YOLO inference and detection
│   └── visualization.py            # Charts and visualizations
│
├── pages/                          # Streamlit pages for different workflows
│   ├── About.py                    # Project information
│   ├── Demo.py                     # Interactive tutorials
│   ├── Training.py                 # Training management UI
│   └── Dataset.py                  # Dataset statistics viewer
│
├── yolo_model/                     # Model storage
│   ├── classes.txt                 # Waste categories
│   └── yolov8_waste_trained.pt     # Your trained model (after training)
│
├── dataset/                        # Training data (after conversion)
│   ├── images/train/               # Training images
│   ├── images/val/                 # Validation images
│   ├── labels/train/               # Training annotations
│   └── labels/val/                 # Validation annotations
│
├── assets/                         # Static files
│   ├── style.css                   # Custom styling
│   └── logo.png                    # App logo
│
└── .streamlit/                     # App configuration
    └── config.toml                 # Theme and settings
```

-----

## 📊 Expected Performance Metrics

After completing the training on the Kaggle dataset, you can expect your model to achieve the following performance metrics.

| Metric | Expected Range | Good Performance |
|---|---|---|
| mAP@0.5 | 0.7 - 0.9 | \> 0.8 |
| Precision | 0.75 - 0.95 | \> 0.85 |
| Recall | 0.7 - 0.9 | \> 0.8 |
| Training Time | 30-90 min | Depends on hardware |

-----

## 💡 Tips for Best Results

  - **Dataset Quality**: For excellent results, aim for a balanced dataset with diverse, high-quality images.
  - **Training Optimization**: Monitor validation loss to avoid overfitting and utilize a GPU for a significant speed boost.
  - **Model Performance**: A mAP \> 0.8 indicates an excellent model. Aim to balance precision and recall for robust performance.

-----

## 🤝 Contributing

We welcome contributions\! Please follow the standard GitHub workflow:

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Open a Pull Request.

-----



## 🙏 Acknowledgments

  - **Ultralytics** for the excellent YOLOv8 implementation
  - **Streamlit** for the amazing web app framework
  - **OpenCV** and **Plotly** for visualization capabilities
  - **PyTorch** for the deep learning infrastructure

-----



