# Sign_Language_Recognition
A complete end-to-end deep learning system for real-time sign language recognition, designed to bridge communication gaps between sign language users and the general population.
# 🎯 Overview
This project implements a sophisticated sign language recognition system that combines Bidirectional Long Short-Term Memory (Bi-LSTM) networks with multi-head attention mechanisms to accurately interpret sign language gestures in real-time. The system processes video sequences through a complete pipeline involving hand detection, feature extraction, temporal modeling, and classification.
# ✨ Key Features
- Interactive Data Collection: Built-in tool for recording custom sign language datasets with real-time visual feedback
- Advanced Deep Learning Architecture:
- Bi-LSTM networks for temporal sequence modeling
- 8-head attention mechanism for focusing on discriminative frames
- MobileNetV2-based feature extraction
- Real-Time Recognition: Processes webcam input at ~30 FPS with 28-35ms inference latency
- Complete Pipeline: Single Python application integrating data collection, training, and inference
- CPU-Optimized: Runs on standard consumer hardware without requiring specialized GPUs

# 📊 System Architecture
- Video Input (30 FPS)
-    ↓
- Hand Detection (HSV Segmentation)
-    ↓
- Feature Extraction (MobileNetV2 → 81-dim)
-    ↓
- Sequence Buffer (30 frames × 81-dim)
-    ↓
- Temporal Modeling (Bi-LSTM: 256 → 128 units)
-    ↓
- Attention Mechanism (8-head + Layer Norm)
-    ↓
- Classification (Softmax Output)
-    ↓
- Predicted Sign + Confidence Score
# 🚀 Getting Started
Prerequisites
- bashPython 3.8+
- TensorFlow 2.20+
- OpenCV 4.12+
- NumPy

- Install required dependencies:

bashpip install tensorflow==2.20 opencv-python numpy
Usage
Run the main application:
bashpython DataCollector.py
```

- The application provides three main options:
#### 1. Data Collection
- Record custom sign language gestures
- Real-time hand detection feedback
- Automatic feature extraction and storage
- Target: 30+ samples per sign for optimal training

#### 2. Model Training
- Automatic 80-20 train-validation split
- Early stopping and learning rate reduction
- Model checkpointing for best weights
- Training completes in 10-30 minutes on CPU

#### 3. Real-Time Recognition
- Live webcam-based sign recognition
- Displays predicted sign with confidence score
- Buffer status and hand detection indicators
- Minimum 50% confidence threshold for predictions

## 📈 Performance Metrics

- **Training Accuracy**: 99.70%
- **Validation Accuracy**: 99.67%
- **Inference Latency**: 28-35ms per prediction
- **Frame Processing Rate**: ~30 FPS
- **Memory Usage**: ~450 MB RAM during inference

## 🏗️ Technical Details

### Model Architecture
- **Feature Extraction**: MobileNetV2 (pretrained on ImageNet) → 81 dimensions
- **Temporal Processing**: 
  - Dense layers (256 → 128 dims) with BatchNorm and Dropout
  - Bi-LSTM (256 units per direction → 128 units per direction)
- **Attention**: 8-head multi-head attention with residual connections
- **Classification**: Dense layers (128 → 64) → Softmax output

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Sparse Categorical Cross-Entropy
- **Batch Size**: 16 sequences
- **Epochs**: Up to 50 (with early stopping)
- **Regularization**: Dropout (30%), Recurrent Dropout (20%)

### Data Format
- **Sequence Length**: 30 frames (~1 second at 30 FPS)
- **Feature Dimension**: 81 per frame
- **Input Shape**: (30, 81)
- **Storage**: Pickled numpy arrays organized by sign class

## 📁 Project Structure
```
- sign-language-translator/
- ├── DataCollector.py          # Main application (data collection, training, inference)
- ├── sign_data/                # Directory for collected sign datasets
- │   ├── sign_1/
- │   ├── sign_2/
- │   └── ...
- ├── best_sign_model.weights.h5   # Trained model weights
- ├── class_names.json          # Sign vocabulary mapping
- └── collection_stats.json     # Dataset statistics
# 🎓 Research Background
This project is based on research in:

- Temporal sequence modeling for gesture recognition
- Attention mechanisms for video understanding
- Real-time computer vision for assistive technology

# Key References

- Bi-LSTM networks for sequential data (Hochreiter & Schmidhuber, 1997)
- Attention mechanisms (Vaswani et al., 2017)
- Sign language recognition surveys (Rastgoo et al., 2021)

# 🔮 Future Enhancements

- Improved Hand Detection: Integration of MediaPipe Hands or OpenPose for robust 3D hand tracking
- Facial Expression Analysis: Capture non-manual markers for complete sign language interpretation
- Expanded Vocabulary: Scale to 100+ signs for practical conversational applications
- Multi-Signer Training: Improve generalization across different signing styles
- Mobile Deployment: TensorFlow Lite conversion for smartphone applications
- Sequence-to-Sequence Translation: Sentence-level translation with grammar correction
- Fingerspelling Recognition: Handle proper names and technical terms
