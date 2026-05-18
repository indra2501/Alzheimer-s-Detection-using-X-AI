"""
Alzheimer's Detection with Explainable AI (XAI)
Implements: LIME, Grad-CAM, Attention Maps, Feature Importance, and Model Interpretability
"""

import numpy as np 
import os
import keras
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, Input
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import cv2
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

# ============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data(max_samples=100):
    """
    Load Alzheimer's MRI images from organized folders
    Returns: image arrays and corresponding labels
    """
    path1, path2, path3, path4 = [], [], [], []
    
    # Windows paths - modify for your system
    base_path = ('C:\\Users\\tomoj\\OneDrive\\Desktop\\Alzheimers\\')
    folders = {
        'Non Demented': path1,
        'Mild Dementia': path2,
        'Moderate Dementia': path3,
        'Very Mild Dementia': path4
    }
    
    for folder_name, path_list in folders.items():
        folder_path = os.path.join(base_path, folder_name)
        if os.path.exists(folder_path):
            for dirname, _, filenames in os.walk(folder_path):
                for filename in filenames:
                    path_list.append(os.path.join(dirname, filename))
    
    # Limit samples for demonstration
    path1 = path1[:max_samples]
    path2 = path2[:max_samples]
    path3 = path3[:max_samples]
    path4 = path4[:max_samples]
    
    return path1, path2, path3, path4

def preprocess_images(path_lists, encoder):
    """
    Preprocess images: resize, normalize, and one-hot encode labels
    """
    data = []
    result = []
    
    for label_idx, paths in enumerate(path_lists):
        for path in paths:
            try:
                img = Image.open(path)
                img = img.resize((128, 128))
                img_array = np.array(img)
                
                # Ensure RGB format
                if len(img_array.shape) == 2:  # Grayscale
                    img_array = np.stack([img_array]*3, axis=-1)
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]
                
                if img_array.shape == (128, 128, 3):
                    data.append(img_array)
                    result.append(encoder.transform([[label_idx]]).toarray())
            except Exception as e:
                print(f"Error processing {path}: {e}")
    
    return np.array(data), np.array(result).reshape((len(data), 4))

# ============================================================================
# SECTION 2: EXPLAINABLE AI - GRAD-CAM
# ============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    Visualizes which regions of the image influenced the model's prediction
    """
    def __init__(self, model, layer_name):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for Grad-CAM features.")

        ensure_model_built(model)
        self.model = model
        self.layer_name = layer_name
        self.grad_model = Model(
            inputs=[model.input],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
    
    def compute_heatmap(self, image, pred_class):
        """
        Compute Grad-CAM heatmap for a given image and class
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(np.expand_dims(image, axis=0))
            loss = predictions[:, pred_class]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        """
        Overlay Grad-CAM heatmap on original image
        """
        heatmap = cv2.resize(heatmap, (128, 128))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        image_normalized = np.uint8(255 * (image / image.max()))
        overlay = cv2.addWeighted(image_normalized, 1 - alpha, heatmap, alpha, 0)
        
        return overlay

# ============================================================================
# SECTION 3: EXPLAINABLE AI - LIME (Local Interpretable Model-agnostic)
# ============================================================================

class SimpleLIME:
    """
    Simplified LIME implementation for image classification
    Explains individual predictions by perturbing image superpixels
    """
    def __init__(self, model, num_superpixels=50, num_samples=1000):
        self.model = model
        self.num_superpixels = num_superpixels
        self.num_samples = num_samples
    
    def segment_image(self, image):
        """
        Simple grid-based image segmentation
        """
        h, w = image.shape[:2]
        seg_h = h // int(np.sqrt(self.num_superpixels))
        seg_w = w // int(np.sqrt(self.num_superpixels))
        
        segmentation = np.zeros((h, w), dtype=int)
        segment_id = 0
        
        for i in range(0, h, seg_h):
            for j in range(0, w, seg_w):
                segmentation[i:i+seg_h, j:j+seg_w] = segment_id
                segment_id += 1
        
        return segmentation
    
    def explain_instance(self, image, predicted_class):
        """
        Explain prediction by analyzing feature importance of superpixels
        """
        segmentation = self.segment_image(image)
        num_segments = segmentation.max() + 1
        
        # Generate perturbed samples
        feature_importance = np.zeros(num_segments)
        
        for _ in range(self.num_samples):
            # Random mask
            mask = np.random.randint(0, 2, num_segments)
            perturbed_image = image.copy()
            
            for seg_id in range(num_segments):
                if mask[seg_id] == 0:
                    perturbed_image[segmentation == seg_id] = 128  # Gray out
            
            # Predict on perturbed image
            pred = self.model.predict(np.expand_dims(perturbed_image, axis=0), verbose=0)
            pred_prob = pred[0][predicted_class]
            
            feature_importance += mask * (pred_prob - 0.5)
        
        feature_importance /= self.num_samples
        return feature_importance, segmentation


def ensure_model_built(model, input_shape=(128, 128, 3)):
    """
    Ensure a Keras model exposes input/output tensors before explainability code uses them.
    """
    try:
        _ = model.input
        return model
    except Exception:
        pass

    dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)

    try:
        model(dummy_input, training=False)
    except Exception:
        try:
            model.build((None,) + input_shape)
        except Exception:
            pass

        model(dummy_input, training=False)

    return model


def prepare_display_image(image):
    """
    Convert an image to an 8-bit RGB array suitable for matplotlib/cv2 display.
    Handles both normalized float images in [0, 1] and already-scaled images in [0, 255].
    """
    image = np.asarray(image)

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype.kind in {'f'}:
        max_value = float(np.max(image)) if image.size else 0.0
        if max_value <= 1.5:
            image = image * 255.0

    return np.clip(image, 0, 255).astype(np.uint8)

# ============================================================================
# SECTION 4: EXPLAINABLE AI - ATTENTION MAPS
# ============================================================================

def compute_attention_map(model, image):
    """
    Create attention-based visualization showing model focus areas
    Uses intermediate layer activations
    """
    ensure_model_built(model)

    # Get intermediate layer outputs
    intermediate_layer_model = Model(
        inputs=model.input,
        outputs=model.get_layer('conv2d_3').output  # Adjust layer name as needed
    )
    
    intermediate_output = intermediate_layer_model.predict(
        np.expand_dims(image, axis=0), verbose=0
    )
    
    # Average across channels
    attention = np.mean(intermediate_output[0], axis=2)
    attention_range = attention.max() - attention.min()
    if attention_range == 0:
        return np.zeros_like(attention)

    attention = (attention - attention.min()) / attention_range
    
    return attention

# ============================================================================
# SECTION 5: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(model, test_images, test_labels, class_names):
    """
    Analyze which image regions are most important for classification
    """
    importance_by_class = {i: [] for i in range(4)}
    
    for image, label in zip(test_images[:20], test_labels[:20]):
        pred = model.predict(np.expand_dims(image, axis=0), verbose=0)
        predicted_class = np.argmax(pred)
        
        # Compute perturbation-based importance
        original_pred = pred[0][predicted_class]
        
        # Create 4x4 grid and measure impact of masking each region
        grid_size = 4
        h, w = 128, 128
        region_h, region_w = h // grid_size, w // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                perturbed = image.copy()
                y1, y2 = i * region_h, (i + 1) * region_h
                x1, x2 = j * region_w, (j + 1) * region_w
                perturbed[y1:y2, x1:x2] = 128  # Gray out region
                
                perturbed_pred = model.predict(np.expand_dims(perturbed, axis=0), verbose=0)
                importance = original_pred - perturbed_pred[0][predicted_class]
                importance_by_class[predicted_class].append(max(0, importance))
    
    return importance_by_class

# ============================================================================
# SECTION 6: MODEL BUILDING AND TRAINING
# ============================================================================

def build_model():
    """
    Build CNN model with named layers for explainability
    """
    inputs = Input(shape=(128, 128, 3), name='input_image')
    x = Conv2D(32, kernel_size=(2, 2), padding='same', name='conv2d_1')(inputs)
    x = Conv2D(32, kernel_size=(2, 2), activation='relu', padding='same', name='conv2d_2')(x)
    x = BatchNormalization(name='batch_norm_1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pool_1')(x)
    x = Dropout(0.25, name='dropout_1')(x)

    x = Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same', name='conv2d_3')(x)
    x = Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same', name='conv2d_4')(x)
    x = BatchNormalization(name='batch_norm_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pool_2')(x)
    x = Dropout(0.25, name='dropout_2')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='dense_1')(x)
    x = Dropout(0.5, name='dropout_3')(x)
    outputs = Dense(4, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='alzheimers_xai_cnn')
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adamax',
        metrics=['accuracy']
    )
    
    return model

# ============================================================================
# SECTION 7: COMPREHENSIVE VISUALIZATION AND REPORTING
# ============================================================================

def create_xai_report(model, image, true_label, class_names):
    """
    Generate comprehensive Explainable AI report for a single prediction
    Includes: prediction confidence, Grad-CAM, attention maps, and feature importance
    """
    ensure_model_built(model)

    # Get prediction
    pred = model.predict(np.expand_dims(image, axis=0), verbose=0)
    predicted_class = np.argmax(pred)
    confidence = pred[0][predicted_class]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(prepare_display_image(image))
    ax1.set_title('Original MRI Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Prediction Confidence
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['#FF6B6B' if i != predicted_class else '#51CF66' for i in range(4)]
    bars = ax2.barh(class_names, pred[0], color=colors)
    ax2.set_xlabel('Confidence Score')
    ax2.set_title('Prediction Confidence', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    
    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, pred[0])):
        ax2.text(val + 0.02, i, f'{val*100:.1f}%', va='center')
    
    # 3. Model Information
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    info_text = f"""
    PREDICTION RESULTS
    {'='*30}
    Predicted: {class_names[predicted_class]}
    True Label: {class_names[true_label]}
    Confidence: {confidence*100:.2f}%
    Match: {'✓ Correct' if predicted_class == true_label else '✗ Incorrect'}
    
    MODEL STATISTICS
    {'='*30}
    Total Parameters: {model.count_params():,}
    Architecture: CNN with 2 Conv blocks
    """
    ax3.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    # 4. Attention Map
    ax4 = fig.add_subplot(gs[1, 0])
    attention = compute_attention_map(model, image)
    im1 = ax4.imshow(attention, cmap='hot')
    ax4.set_title('Attention Map\n(Activation Intensity)', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im1, ax=ax4, fraction=0.046, pad=0.04)
    
    # 5. Feature Importance Grid
    ax5 = fig.add_subplot(gs[1, 1])
    importance_grid = np.zeros((4, 4))
    original_pred = pred[0][predicted_class]
    
    grid_size = 4
    for i in range(grid_size):
        for j in range(grid_size):
            perturbed = image.copy()
            y1, y2 = i * 32, (i + 1) * 32
            x1, x2 = j * 32, (j + 1) * 32
            perturbed[y1:y2, x1:x2] = 128
            
            perturbed_pred = model.predict(np.expand_dims(perturbed, axis=0), verbose=0)
            importance = original_pred - perturbed_pred[0][predicted_class]
            importance_grid[i, j] = max(0, importance)
    
    im2 = ax5.imshow(importance_grid, cmap='YlOrRd')
    ax5.set_title('Region Importance Score\n(Higher = More Important)', 
                  fontsize=12, fontweight='bold')
    ax5.set_xticks(range(4))
    ax5.set_yticks(range(4))
    plt.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04)
    
    # 6. Overlay: Original + Attention
    ax6 = fig.add_subplot(gs[1, 2])
    attention_resized = cv2.resize(attention, (128, 128))
    attention_colored = cv2.applyColorMap(np.uint8(255 * attention_resized), cv2.COLORMAP_JET)
    image_normalized = prepare_display_image(image)
    overlay = cv2.addWeighted(image_normalized, 0.6, attention_colored, 0.4, 0)
    ax6.imshow(overlay)
    ax6.set_title('Attention Overlay\n(Model Focus Areas)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # 7. Prediction Distribution by Region
    ax7 = fig.add_subplot(gs[2, :])
    region_predictions = []
    region_labels = []
    
    for i in range(4):
        for j in range(4):
            perturbed = image.copy()
            y1, y2 = i * 32, (i + 1) * 32
            x1, x2 = j * 32, (j + 1) * 32
            perturbed[y1:y2, x1:x2] = 128
            
            perturbed_pred = model.predict(np.expand_dims(perturbed, axis=0), verbose=0)
            region_predictions.append(perturbed_pred[0])
            region_labels.append(f'R{i}{j}')
    
    region_predictions = np.array(region_predictions)
    x_pos = np.arange(len(region_labels))
    
    for class_idx in range(4):
        ax7.bar(x_pos + class_idx*0.2, region_predictions[:, class_idx], 
               width=0.2, label=class_names[class_idx], alpha=0.8)
    
    ax7.set_xlabel('Image Region')
    ax7.set_ylabel('Prediction Confidence')
    ax7.set_title('Prediction Changes When Each Region is Masked', 
                  fontsize=12, fontweight='bold')
    ax7.set_xticks(x_pos + 0.3)
    ax7.set_xticklabels(region_labels, rotation=45)
    ax7.legend(loc='upper right')
    ax7.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Explainable AI Report - Alzheimer\'s Detection', 
                fontsize=16, fontweight='bold', y=0.995)
    
    return fig

# ============================================================================
# SECTION 8: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    if not TF_AVAILABLE:
        print("Warning: TensorFlow not found. Grad-CAM features will be limited.")
    
    class_names = ['Non Demented', 'Mild Dementia', 'Moderate Dementia', 'Very Mild Dementia']
    
    # Load and preprocess data
    print("Loading data...")
    path1, path2, path3, path4 = load_data(max_samples=100)
    
    encoder = OneHotEncoder()
    encoder.fit([[0], [1], [2], [3]])
    
    print("Preprocessing images...")
    data, result = preprocess_images([path1, path2, path3, path4], encoder)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Labels shape: {result.shape}")
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        data, result, test_size=0.15, shuffle=True, random_state=42
    )
    
    # Normalize pixel values
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # Build and train model
    print("\nBuilding model...")
    model = build_model()
    ensure_model_built(model)
    print(model.summary())
    
    print("\nTraining model...")
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=10,
        verbose=1,
        validation_data=(x_test, y_test)
    )
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss Over Epochs', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy Over Epochs', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate XAI reports for test samples
    print("\nGenerating Explainable AI reports...")
    for idx in range(min(3, len(x_test))):
        test_image = x_test[idx]
        true_label = np.argmax(y_test[idx])
        
        fig = create_xai_report(model, test_image, true_label, class_names)
        plt.savefig(f'xai_report_{idx}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("\nExplainable AI implementation complete!")
    print("Generated reports:")
    print("  - xai_report_0.png, xai_report_1.png, xai_report_2.png")
    print("  - training_history.png")

