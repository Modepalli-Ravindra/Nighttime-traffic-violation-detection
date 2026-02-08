# Nighttime Traffic Violation Detection System

## 📌 Overview
This project is an **AI-powered Traffic Violation Detection System** specialized for **nighttime conditions**. It processes video footage to detect traffic violations such as **Signal Jumping** and **Triple Riding** using Computer Vision and Deep Learning techniques.

The system utilizes **YOLOv8** for object detection and tracking, combined with image enhancement techniques (Gamma Correction & CLAHE) to improve visibility in low-light scenarios.

## 🚀 Features
- **Nighttime Visibility Enhancement**: Uses Gamma Correction and CLAHE (Contrast Limited Adaptive Histogram Equalization) to assist detection in dark environments.
- **Signal Jump Detection**: Identifies vehicles crossing the stop line during a red light.
  - *Note: Includes a simulated traffic light logic for demonstration purposes.*
- **Triple Riding Detection**: Detects motorcycles with more than two riders using both heuristic analysis (person-bike overlap) and a custom trained model.
- **Helmet Detection**: (Experimental) Identifies riders without helmets.
- **Web Dashboard**: A user-friendly Flask-based web interface to upload videos and view processed results with real-time statistics.

## 🛠️ Tech Stack
- **Backend Framework**: Python (Flask)
- **Computer Vision**: OpenCV, scikit-image
- **Deep Learning**: Ultralytics YOLOv8
- **Frontend**: HTML5, CSS3, JavaScript

## 📂 Project Structure
```
├── app.py                 # Main Flask application
├── requirements.txt       # Project dependencies
├── src/
│   └── detector.py        # Core detection logic (YOLO + Image Processing)
├── static/                # CSS, JS, and images
├── templates/             # HTML templates
├── weights/               # Pre-trained YOLO weights
└── input_videos/          # Directory for uploading videos
```

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Modepalli-Ravindra/Nighttime-traffic-violation-detection.git
   cd Nighttime-traffic-violation-detection
   ```

2. **Create & Activate Virtual Environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```
   The app will start at `http://127.0.0.1:5000/`.

## 🧠 How It Works
1. **Video Input**: Users upload a traffic video via the dashboard.
2. **Preprocessing**: Frames are enhanced to improve clarity in low-light conditions.
3. **Detection & Tracking**:
   - `yolov8n.pt` tracks vehicles (Cars, Bikes, Trucks, etc.) and alignment.
   - Analysis logic checks for line crossing (Signal Jump) and rider counts (Triple Riding).
   - A custom model (if available) further validates specific violation types.
4. **Output**: The processed video is displayed with bounding boxes and violation labels, alongside a live counter.

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a Pull Request.

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
