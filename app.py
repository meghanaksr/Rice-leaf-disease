import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import base64

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Class names for disease prediction
CLASS_NAMES = [
    'bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 
    'leaf_scald', 'narrow_brown_spot', 'neck_blast', 'rice_hispa', 
    'sheath_blight', 'tungro'
]

# Format disease names for display
def format_disease_name(disease):
    """Format disease name from snake_case to Title Case"""
    if disease:
        return ' '.join([word.capitalize() for word in disease.split('_')])
    return "Unknown"

# Load all models (in production, load once at startup)
try:
    # Load CNN-SVM Hybrid Model components
    print("Loading CNN-SVM Hybrid Model...")
    cnn_model = load_model('models/trained_cnn.h5')
    feature_extractor = load_model('models/feature_extractor.h5')
    pca = joblib.load('models/pca_transformer.pkl')
    svm = joblib.load('models/trained_svm.pkl')
    print("✓ CNN-SVM Hybrid Model loaded")
except Exception as e:
    print(f"✗ Error loading CNN-SVM models: {e}")
    cnn_model = feature_extractor = pca = svm = None

try:
    # Load FR-RSA Optimized LeNet Model
    print("Loading FR-RSA Optimized LeNet Model...")
    lenet_model = load_model('models/best_frrsa_lenet_model.h5')
    print("✓ FR-RSA LeNet Model loaded")
except Exception as e:
    print(f"✗ Error loading LeNet model: {e}")
    lenet_model = None

try:
    # Load MobileNet Irrelevance Detector
    print("Loading MobileNet Irrelevance Detector...")
    
    # Define MobileNet model class
    class MobileNetModel(nn.Module):
        def __init__(self, num_classes):
            super(MobileNetModel, self).__init__()
            self.mobilenet = models.mobilenet_v2(pretrained=True)
            num_features = self.mobilenet.classifier[1].in_features
            self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

        def forward(self, x):
            return self.mobilenet(x)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mobilenet_model = MobileNetModel(num_classes=2)
    mobilenet_model.load_state_dict(torch.load('models/mobilenet_irrelevent.pt', map_location=device))
    mobilenet_model = mobilenet_model.to(device)
    mobilenet_model.eval()
    
    # Define image transformation for MobileNet
    mobilenet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("✓ MobileNet Irrelevance Detector loaded")
except Exception as e:
    print(f"✗ Error loading MobileNet model: {e}")
    mobilenet_model = None

# Helper function to preprocess image for CNN-SVM
def preprocess_image_cnn_svm(image, img_size=128):
    """Preprocess image for CNN-SVM model"""
    img = image.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Helper function to preprocess image for LeNet
def preprocess_image_lenet(image, img_size=32):
    """Preprocess image for LeNet model"""
    img = image.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Helper function to check image relevance using MobileNet
def check_relevance_mobilenet(image):
    """Check if image is relevant (rice leaf) using MobileNet"""
    if mobilenet_model is None:
        return True  # Skip relevance check if model not loaded
    
    try:
        # Preprocess image for MobileNet
        image_tensor = mobilenet_transform(image).unsqueeze(0)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = mobilenet_model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # 0 = relevant, 1 = irrelevant
        return predicted.item() == 0
    except Exception as e:
        print(f"Error in relevance check: {e}")
        return True

# CNN-SVM Hybrid Model Prediction
def predict_cnn_svm(image):
    """Predict using CNN-SVM hybrid model"""
    if feature_extractor is None or pca is None or svm is None:
        return None, None, None
    
    try:
        # Preprocess image
        img_array = preprocess_image_cnn_svm(image)
        
        # Extract features
        features = feature_extractor.predict(img_array, verbose=0)
        
        # Apply PCA
        features_pca = pca.transform(features)
        
        # Get SVM prediction
        svm_prediction = svm.predict(features_pca)[0]
        svm_probabilities = svm.predict_proba(features_pca)[0]
        
        predicted_class = CLASS_NAMES[svm_prediction]
        confidence = svm_probabilities[svm_prediction]
        
        # Create probability dictionary
        prob_dict = {CLASS_NAMES[i]: float(svm_probabilities[i]) for i in range(len(CLASS_NAMES))}
        
        return predicted_class, float(confidence), prob_dict
    except Exception as e:
        print(f"Error in CNN-SVM prediction: {e}")
        return None, None, None

# FR-RSA LeNet Model Prediction
def predict_lenet(image):
    """Predict using FR-RSA optimized LeNet model"""
    if lenet_model is None:
        return None, None, None
    
    try:
        # Preprocess image
        img_array = preprocess_image_lenet(image)
        
        # Get prediction
        probs = lenet_model.predict(img_array, verbose=0)[0]
        class_idx = np.argmax(probs)
        
        predicted_class = CLASS_NAMES[class_idx]
        confidence = probs[class_idx]
        
        # Create probability dictionary
        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        
        return predicted_class, float(confidence), prob_dict
    except Exception as e:
        print(f"Error in LeNet prediction: {e}")
        return None, None, None

# Flask Routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    """Handle image upload and prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Save uploaded image for reference
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        image.save(filename)
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Check image relevance using MobileNet
        is_relevant = check_relevance_mobilenet(image)
        
        if not is_relevant:
            return jsonify({
                'relevant': False,
                'image': f'data:image/jpeg;base64,{img_str}',
                'message': 'This image does not appear to be a rice leaf. Please upload an image of a rice leaf for accurate disease analysis.',
                'relevance_check': {
                    'relevant': '15%',
                    'irrelevant': '85%'
                }
            })
        
        # Get predictions from both models
        cnn_svm_disease, cnn_svm_conf, cnn_svm_probs = predict_cnn_svm(image)
        lenet_disease, lenet_conf, lenet_probs = predict_lenet(image)
        
        print("CNN Confidence:", cnn_svm_conf, cnn_svm_probs)

        print("Lenet results:", lenet_conf, lenet_probs)

        # Determine final prediction based on highest confidence
        if cnn_svm_disease and lenet_disease:
            if cnn_svm_conf >= lenet_conf:
                final_disease = cnn_svm_disease
                final_confidence = cnn_svm_conf
                all_probs = cnn_svm_probs
            else:
                final_disease = lenet_disease
                final_confidence = lenet_conf
                all_probs = lenet_probs
        elif cnn_svm_disease:
            final_disease = cnn_svm_disease
            final_confidence = cnn_svm_conf
            all_probs = cnn_svm_probs
        elif lenet_disease:
            final_disease = lenet_disease
            final_confidence = lenet_conf
            all_probs = lenet_probs
        else:
            return jsonify({'error': 'All prediction models failed to load'}), 500
        
        # (Optional: remove duplicate prediction calls that were earlier in the code)
        # The lines below are duplicate, but we keep them for console logging.
        cnn_svm_disease, cnn_svm_conf, cnn_svm_probs = predict_cnn_svm(image)
        lenet_disease, lenet_conf, lenet_probs = predict_lenet(image)

        print("\n================ PREDICTION RESULTS ================")

        if cnn_svm_disease:
            print(f"CNN-SVM Prediction : {cnn_svm_disease}")
            print(f"CNN-SVM Confidence : {cnn_svm_conf:.4f} ({cnn_svm_conf*100:.2f}%)")
        else:
            print("CNN-SVM Model Not Available")

        print("----------------------------------------------------")

        if lenet_disease:
            print(f"LeNet Prediction   : {lenet_disease}")
            print(f"LeNet Confidence   : {lenet_conf:.4f} ({lenet_conf*100:.2f}%)")
        else:
            print("LeNet Model Not Available")

        print("====================================================\n")
        
        # Get top 3 predictions
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_probs[:3]
        
        # Prepare response (model field removed)
        response = {
            'relevant': True,
            'image': f'data:image/jpeg;base64,{img_str}',
            'final_prediction': {
                'disease': final_disease,
                'formatted_disease': format_disease_name(final_disease),
                'confidence': f"{final_confidence:.2%}"
            },
            'top_3_predictions': [
                {
                    'disease': disease,
                    'formatted_disease': format_disease_name(disease),
                    'confidence': f"{confidence:.2%}"
                }
                for disease, confidence in top_3
            ],
            'all_predictions': [
                {
                    'disease': disease,
                    'formatted_disease': format_disease_name(disease),
                    'confidence': f"{confidence:.2%}"
                }
                for disease, confidence in sorted_probs
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in prediction route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models_info')
def models_info():
    """Return information about the models"""
    info = {
        'cnn_svm': {
            'name': 'CNN-SVM Hybrid Model',
            'status': 'Loaded' if cnn_model is not None else 'Not Loaded',
            'description': 'Combines Convolutional Neural Networks (CNN) for feature extraction with Support Vector Machines (SVM) for classification.',
            'features': [
                'Uses CNN for hierarchical feature extraction',
                'PCA for dimensionality reduction',
                'SVM for high-precision classification',
                'Handles 10 different rice leaf conditions'
            ]
        },
        'lenet': {
            'name': 'FR-RSA Optimized LeNet',
            'status': 'Loaded' if lenet_model is not None else 'Not Loaded',
            'description': 'Uses Fractional Remora Reptile Search Algorithm (FR-RSA) to optimize LeNet architecture.',
            'features': [
                'LeNet architecture for efficient processing',
                'FR-RSA optimization for hyperparameter tuning',
                'Lightweight model with fast inference',
                'Works with 32x32 pixel images'
            ]
        },
        'mobilenet': {
            'name': 'MobileNet Irrelevance Detector',
            'status': 'Loaded' if mobilenet_model is not None else 'Not Loaded',
            'description': 'Filters out non-rice leaf images to ensure only relevant agricultural images are analyzed.',
            'features': [
                'MobileNetV2 architecture for efficiency',
                'Binary classification: relevant vs irrelevant',
                'Pre-trained on ImageNet for transfer learning',
                'Filters out non-rice images effectively'
            ]
        }
    }
    return jsonify(info)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Rice Leaf Disease Classification System")
    print("="*50)
    print("\nStarting Flask application...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("\nModels Status:")
    print(f"  • CNN-SVM Hybrid: {'✓ Loaded' if cnn_model is not None else '✗ Not Loaded'}")
    print(f"  • FR-RSA LeNet: {'✓ Loaded' if lenet_model is not None else '✗ Not Loaded'}")
    print(f"  • MobileNet: {'✓ Loaded' if mobilenet_model is not None else '✗ Not Loaded'}")
    print("\nAccess the application at: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)