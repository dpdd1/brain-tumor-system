# Intelligent Brain Tumor Diagnostic Analysis System

## Introduction

The Intelligent Brain Tumor Diagnostic Analysis System is a deep learning-based medical imaging analysis platform that provides functions for brain tumor classification, detection, segmentation, and video diagnosis. The system uses the Flask framework to build web applications, integrates multiple AI models, and provides intelligent auxiliary diagnostic tools for medical institutions and doctors.

## Feature Highlights

### User Authentication Module
- **User Registration**: Supports custom avatar selection (multiple doctor avatars available)
- **User Login**: Secure authentication mechanism
- **User Logout**: Clears session information
- **Profile Management**: Supports modification of basic information and password

### Imaging Diagnosis Module
- **Brain Tumor Classification**: Uses classification models to identify tumor types
- **Tumor Detection**: Locates tumor regions within images
- **Result Display**: Shows diagnosis results, confidence levels, and suggestions
- **Image Comparison**: Displays comparison between original images and diagnosis results

### Image Segmentation Module
- **Tumor Segmentation**: Precisely segments brain tumor regions
- **Parameter Adjustment**: Supports adjustment of Gaussian blur, contrast, and threshold parameters
- **Interactive Preview**: View segmentation effects in real-time
- **Save Segmentation Results**: Export segmented images

### Video Diagnosis Module
- **Real-time Video Stream Processing**
- **Frame-by-frame Video Analysis**
- **Video Control**: Start, pause, stop detection
- **Detection Result Logging and Export**

### Medical Record Management
- **Record Keeping**: Saves diagnosis records
- **Record Query**: Supports search and filtering
- **Record Details**: View complete diagnosis information
- **Batch Operations**: Batch delete records
- **PDF Export**: Generate diagnosis report PDF

### Data Analysis Module
- **Data Statistics**: Display statistical information on diagnosis data
- **Chart Visualization**: Use charts to display diagnosis trends
- **Data Export**: Supports data export functionality

## Tech Stack

- **Backend Framework**: Flask
- **Deep Learning Framework**: PyTorch
- **Database**: SQLite (users.db)
- **Frontend Technologies**: HTML5, CSS3, JavaScript
- **Chart Library**: Chart.js
- **AI Models**:
  - classification.pt: Classification model
  - detection.pt: Detection model
  - segmentation.pt: Segmentation model

## Project Structure

```
brain-tumor-system/
├── app.py                 # Main Flask application file
├── models/
│   ├── classification.pt  # Classification model
│   ├── detection.pt       # Detection model
│   └── segmentation.pt    # Segmentation model
├── static/
│   ├── css/
│   │   ├── style.css      # Main style file
│   │   └── styles.css     # Additional styles
│   ├── js/
│   │   ├── app.js         # Main application logic
│   │   ├── records.js     # Medical record management script
│   │   ├── script.js      # General scripts
│   │   └── particles-config.js  # Particle effect configuration
│   ├── images/            # SVG icon resources
│   └── avatars/           # User avatar storage
├── templates/
│   ├── login.html         # Login page
│   ├── register.html      # Registration page
│   ├── diagnosis.html     # Diagnosis page
│   ├── segmentation.html  # Segmentation page
│   ├── records.html       # Medical record page
│   ├── video.html         # Video diagnosis page
│   ├── profile.html       # Profile page
│   ├── analysis.html      # Analysis page
│   └── data_analysis.html # Data analysis page
├── requirements.txt       # Python dependencies
└── users.db              # SQLite database
```

## Installation and Deployment

### Environment Requirements
- Python 3.7+
- CUDA (Recommended, for GPU acceleration)

### Installation Steps

1. Clone the project locally:
```bash
git clone https://gitee.com/dpdd01/brain-tumor-system.git
cd brain-tumor-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the application:
```bash
python app.py
```

5. Access the system:
Open your browser and visit http://localhost:5000

## User Guide

### User Registration and Login
1. Visit the system homepage and click the Register button
2. Select a preferred doctor avatar
3. Fill in username, password, and other information to complete registration
4. Log in to the system using the registered account

### Imaging Diagnosis
1. Enter the diagnosis page after logging in
2. Click the upload area to select a brain image
3. The system automatically analyzes and displays diagnosis results
4. View classification results, confidence levels, and suggestion information

### Image Segmentation
1. Enter the segmentation page
2. Upload the original medical image
3. Adjust parameters using sliders:
   - Threshold: Adjust segmentation threshold
   - Gaussian Blur: Noise reduction processing
   - Contrast: Adjust image contrast
4. View segmentation effects in real-time, save results when satisfied

### Video Diagnosis
1. Enter the video diagnosis page
2. Click the start button to launch the camera
3. The system analyzes every frame in the video stream in real-time
4. Pause or stop detection to view results
5. Export detection report

### Medical Record Management
1. Click "Records" in the navigation bar to enter the medical record page
2. Use the search box to find specific records
3. Click a record to view detailed information
4. Editing, deleting, and other operations can be performed
5. Supports batch selection and deletion

### Data Analysis
1. Enter the data analysis page to view statistical charts
2. View diagnosis data distribution and trends
3. Supports data refresh and export

## Interface Features

- **Animated Background**: Dynamic particle background effect
- **Responsive Design**: Adapts to different screen sizes
- **Smooth Animation**: Page switching and element interaction animations
- **Dark Theme**: Professional medical software interface style
- **Visual Components**: Data charts and progress bars

## Notes

1. This system is for auxiliary diagnosis use only; final diagnosis results should be confirmed by a professional doctor
2. Ensure uploaded medical images meet system requirements (Supported formats: PNG, JPG, JPEG)
3. It is recommended to run in a GPU environment for better performance
4. Please safeguard your account password properly and do not leak it to others

## License

This project follows an open-source license agreement. For specific information, please refer to the LICENSE file.

## Technical Support

If you have any questions or suggestions, please contact the project maintainer through the Gitee platform.

---

*Intelligent Brain Tumor Diagnostic Analysis System - Empowering Medical Diagnosis with AI*