# Indian Mammal Species Identifier

## Overview

This is a Flask-based web application that identifies mammal species from the Indian subcontinent using a two-stage machine learning pipeline. The system first detects whether an uploaded image contains a mammal, then classifies the specific species using either a custom-trained ResNet50 model or Google's Gemini AI as a fallback. The application supports common image formats and provides a clean, user-friendly web interface for image upload and species identification.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Technology Stack**: HTML5 with embedded CSS and JavaScript
- **Design Pattern**: Single-page application with drag-and-drop file upload
- **Key Features**:
  - Responsive design with dark theme UI
  - Drag-and-drop and click-to-browse file upload
  - Real-time image preview and loading states
  - Collapsible tips section for user guidance
  - Support for PNG, JPG, JPEG, and WebP formats

### Backend Architecture

#### Web Framework
- **Framework**: Flask (Python)
- **Design Pattern**: RESTful API with template rendering
- **Key Routes**:
  - `/` - Serves the main HTML interface
  - `/predict` - POST endpoint for image classification
  - File serving for uploaded media

#### Three-Stage ML Pipeline

**Stage 1: Mammal Detection**
- **Model**: EfficientNet-B0 based binary classifier
- **Purpose**: Validates that uploaded image contains a mammal
- **Architecture**: Custom classifier head with BatchNorm, ReLU, and Dropout layers
- **Threshold**: Configurable confidence threshold (default: 0.8)
- **Rationale**: Reduces false positives by filtering non-mammal images before species classification
- **Output**: If not a mammal, pipeline stops and returns "Not a Mammal"

**Stage 2: Species Classification (F1 Model)**
- **Model**: ResNet50 fine-tuned on 74 mammal species
- **Confidence Threshold**: 0.6 (60%)
- **Purpose**: Primary species identification with high accuracy
- **Output**: If confident, returns species name from F1 model

**Stage 3: Gemini AI Fallback**
- **Trigger**: When F1 model confidence is below 0.6
- **Model**: Google Gemini 2.5 Flash API
- **Design Decision**: Hybrid approach provides robustness - local ML model for speed, cloud AI for edge cases
- **Species Scope**: Limited to Indian subcontinent mammals (India, Pakistan, Bangladesh, Sri Lanka, Nepal, Bhutan)
- **Output**: Species name from Gemini with indication that F1 model had low confidence

#### Image Processing Pipeline
- **Input Validation**: File extension whitelist (png, jpg, jpeg, webp)
- **Preprocessing**: 
  - Automatic RGB conversion
  - JPEG conversion for consistent format
  - Resize to 224x224 pixels
  - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Storage**: Local media folder with secure filename handling
- **Size Limit**: 200MB maximum upload size

#### Model Management
- **Loading Strategy**: Models loaded once at application startup
- **Device Selection**: Automatic GPU/CPU detection via PyTorch
- **Model Paths**: 
  - Mammal detector: `models/mammal_detector.pth`
  - Species classifier: Uses checkpoint loading with state dict inspection
  - Species labels: `models/species_classes.json`

### Data Storage Solutions
- **File Storage**: Local filesystem (`media/` directory)
- **Model Artifacts**: PyTorch `.pth` checkpoint files
- **Configuration**: JSON file for species class names (170 species)
- **No Database**: Application is stateless with no persistent data storage beyond uploaded files

### External Dependencies

#### Third-Party ML Libraries
- **PyTorch**: Deep learning framework for model inference
- **TorchVision**: Pre-trained models (EfficientNet-B0, ResNet50) and image transformations
- **PIL (Pillow)**: Image loading, conversion, and preprocessing

#### Cloud AI Services
- **Google Gemini AI**: 
  - Model: gemini-2.5-flash
  - Purpose: Fallback species identification when local model confidence is low
  - API Key: Environment variable `GEMINI_API_KEY`
  - Input: JPEG encoded images with expert prompt engineering
  - Output: Scientific species names in Genus_species format

#### Web Framework & Utilities
- **Flask**: Web server and routing
- **Werkzeug**: Secure filename handling and file utilities

#### Training Infrastructure (Development Only)
- **Data Augmentation**: Extensive transforms for model training
- **Training Tools**: 
  - Mixed precision training (torch.cuda.amp)
  - Learning rate scheduling (ReduceLROnPlateau, CosineAnnealingLR)
  - WeightedRandomSampler for class imbalance
  - Early stopping mechanism
- **Visualization**: Matplotlib, Seaborn for training metrics

#### Model Training Considerations
- **Class Imbalance Handling**: Weighted loss function and sampling strategy
- **Regularization**: Dropout layers, data augmentation, early stopping
- **Optimization**: Adam optimizer with adaptive learning rate scheduling
- **Checkpointing**: Best model saving with resume capability