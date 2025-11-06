# Final Fully-Commented Code for Jupyter Notebook
## Copy these cells into your notebook

---

## Cell 1: Header Comments (REQUIRED)
```python
# ============================================================
# CS4287 Neural Computing - Assignment 2
# Convolutional Neural Networks: Transfer Learning with ResNet50
# ============================================================
#
# Team Members:
# Todor Aleksandrov - 22336303
# Darragh Kennedy - 22346945
#
# Code Execution Status: This code executes to completion without errors
# when run in Google Colab with GPU enabled and correct dataset paths.
#
# Third Party Source: ResNet50 architecture from Keras Applications
# https://keras.io/api/applications/resnet/#resnet50-function
# Pre-trained weights from ImageNet dataset
#
# Dataset Source: Kaggle Fruit Detection Dataset
# https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection
# Converted from YOLO format to classification format
#
# ============================================================
```

---

## Cell 2: Imports and Setup
```python
# ============================================================
# SECTION 1: IMPORTS AND ENVIRONMENT SETUP
# PDF Report Reference: Introduction and Setup
# ============================================================

# Standard Python libraries for data manipulation and visualization
import numpy as np              # Numerical operations on arrays/tensors
import pandas as pd            # Data structures for results tracking
import matplotlib.pyplot as plt  # Plotting and visualization
import seaborn as sns          # Enhanced statistical visualizations
import os                      # File system operations
from datetime import datetime  # Training time tracking
import warnings
warnings.filterwarnings('ignore')  # Suppress TensorFlow warnings for cleaner output

# Deep Learning Framework - TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,  # Core layers
    RandomFlip, RandomRotation, RandomZoom       # Data augmentation layers
)
from tensorflow.keras.applications import ResNet50  # Pre-trained ResNet50
from tensorflow.keras.optimizers import Adam  # Optimizer - Section 4
from tensorflow.keras.callbacks import (
    EarlyStopping,         # Stop training when no improvement
    ReduceLROnPlateau,     # Reduce learning rate on plateau
    ModelCheckpoint        # Save best model during training
)
from tensorflow.keras.metrics import (
    TopKCategoricalAccuracy,  # Top-K accuracy metric
    Precision,                # Precision metric
    Recall                    # Recall metric
)

# Scikit-learn for evaluation and cross-validation
from sklearn.model_selection import StratifiedKFold  # K-fold CV with stratification
from sklearn.metrics import (
    classification_report,   # Per-class precision/recall/F1
    confusion_matrix,        # Confusion matrix for error analysis
    precision_recall_fscore_support  # Detailed metrics
)

# Set random seeds for reproducibility across runs
# This ensures consistent results when re-running experiments
np.random.seed(42)
tf.random.set_seed(42)

# Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

# Print versions for documentation
print("=" * 70)
print("ENVIRONMENT INFORMATION")
print("=" * 70)
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {os.sys.version}")
print("=" * 70)
```

---

## Cell 3: Data Loading
```python
# ============================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# PDF Report Reference: Section 1 (The Dataset)
# ============================================================

# Dataset Configuration
# Path to preprocessed fruit classification dataset on Google Drive
DATASET_PATH = '/content/drive/MyDrive/CS4287_Assignment/data/fruits_classification'
IMAGE_SIZE = (224, 224)  # ResNet50 requires 224x224 input images
BATCH_SIZE = 32          # Number of images processed simultaneously
                         # Balances memory usage and gradient quality

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)
print(f"Dataset path: {DATASET_PATH}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print("=" * 70)

# Load Training Dataset
# Keras automatically creates one-hot encoded labels from folder structure
# Each subfolder name becomes a class label (Apple, Banana, Grape, etc.)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f'{DATASET_PATH}/train',
    image_size=IMAGE_SIZE,      # Resize all images to 224x224
    batch_size=BATCH_SIZE,      # Group images into batches
    label_mode='categorical',    # One-hot encode labels for multi-class
    shuffle=True,                # Shuffle for better training
    seed=42                      # Reproducible shuffling
)

# Load Test Dataset (for final evaluation - Section 6)
# shuffle=False to maintain consistent order for analysis
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f'{DATASET_PATH}/test',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False,               # Keep consistent order for evaluation
    seed=42
)

# Load Validation Dataset (for hyperparameter tuning - Section 8)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f'{DATASET_PATH}/valid',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False,               # Keep consistent order for evaluation
    seed=42
)

# Extract class names from folder structure
# These are automatically sorted alphabetically by Keras
class_names = train_ds.class_names
num_classes = len(class_names)

# Calculate dataset sizes
num_train_batches = tf.data.experimental.cardinality(train_ds).numpy()
num_test_batches = tf.data.experimental.cardinality(test_ds).numpy()
num_val_batches = tf.data.experimental.cardinality(val_ds).numpy()

# Approximate sample counts (batches × batch_size)
train_samples = num_train_batches * BATCH_SIZE
test_samples = num_test_batches * BATCH_SIZE
val_samples = num_val_batches * BATCH_SIZE

print(f"\n✓ Dataset loaded successfully!")
print(f"  Classes ({num_classes}): {class_names}")
print(f"  Training samples: ~{train_samples} ({num_train_batches} batches)")
print(f"  Validation samples: ~{val_samples} ({num_val_batches} batches)")
print(f"  Test samples: ~{test_samples} ({num_test_batches} batches)")
print(f"  Input shape: {IMAGE_SIZE + (3,)}")  # (224, 224, 3) for RGB
print("=" * 70)
```

---

## Cell 4: Data Visualization
```python
# ============================================================
# SECTION 1.3: DATA VISUALIZATION
# PDF Report Reference: Section 1 (The Dataset)
# Generates Figure 1 (class distribution) and Figure 2 (sample images)
# ============================================================

print("\nGenerating dataset visualizations...")

# ============================================================
# Figure 2: Sample Fruit Images
# ============================================================
plt.figure(figsize=(15, 15))

# Take first batch from training data for visualization
for images, labels in train_ds.take(1):
    # Display grid of 16 sample images
    for i in range(min(16, len(images))):
        plt.subplot(4, 4, i + 1)
        
        # Convert tensor to displayable image
        # images[i] is in range [0, 1], convert to [0, 255] for display
        img_array = images[i].numpy()
        img_array = (img_array * 255).astype("uint8")  # Scale to 0-255
        plt.imshow(img_array)
        
        # Get class name from one-hot encoded label
        # labels[i] is [0,0,1,0,0,0] format, argmax finds the 1
        label_idx = np.argmax(labels[i].numpy())
        class_name = class_names[label_idx]
        plt.title(f"{class_name}", fontsize=12, fontweight='bold')
        plt.axis('off')  # Hide axis for cleaner look

plt.suptitle('Sample Fruits from Training Set', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_fruit.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sample_fruit.png (Figure 2 in PDF)")
plt.show()

# ============================================================
# Figure 1: Class Distribution Bar Chart
# ============================================================
# Count samples per class by iterating through entire training dataset
class_counts = {name: 0 for name in class_names}

# Iterate through all batches to count samples
for images, labels in train_ds:
    for label in labels:
        # Get class index from one-hot encoding
        idx = np.argmax(label.numpy())
        class_name = class_names[idx]
        class_counts[class_name] += 1

# Create bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(class_counts.keys(), class_counts.values(), 
               color='steelblue', edgecolor='black', linewidth=1.5)
plt.title('Distribution of Fruit Classes in Training Set', 
          fontsize=14, fontweight='bold')
plt.xlabel('Fruit Class', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add count labels on top of bars for exact values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('bar chart.png', dpi=300, bbox_inches='tight')
print("✓ Saved: bar chart.png (Figure 1 in PDF)")
plt.show()

# Print class distribution with percentages
# Identifies class imbalance (important for Section 1.2)
total_samples = sum(class_counts.values())
print("\n" + "=" * 70)
print("CLASS DISTRIBUTION ANALYSIS")
print("=" * 70)
for class_name in sorted(class_names):
    count = class_counts[class_name]
    percentage = (count / total_samples) * 100
    print(f"  {class_name:12s}: {count:5d} images ({percentage:5.1f}%)")

# Calculate imbalance ratio (max class / min class)
max_count = max(class_counts.values())
min_count = min(class_counts.values())
imbalance_ratio = max_count / min_count
print(f"\n  Class imbalance ratio: {imbalance_ratio:.2f}:1")
print(f"  (Largest class / Smallest class)")
print("=" * 70)
```

---

## Cell 5: Model Building Function
```python
# ============================================================
# SECTION 2: NETWORK ARCHITECTURE - TRANSFER LEARNING
# PDF Report Reference: Section 2 (Network Structure)
# ============================================================

def build_resnet50_model_with_dropout(dropout_rates, learning_rate=0.001):
    """
    Build ResNet50 transfer learning model for fruit classification.
    
    Transfer Learning Strategy (Section 2.2):
    ----------------------------------------
    - Use pre-trained ResNet50 as frozen feature extractor
    - ResNet50 trained on ImageNet (1.4M images, 1000 classes)
    - Add custom classification head for our 6 fruit classes
    - Only train classification head (1.2M params vs 24M total)
    
    ResNet50 Architecture (He et al., 2015):
    ---------------------------------------
    - 50 layers deep with residual (skip) connections
    - Residual block: y = F(x) + x (identity shortcut)
    - Solves vanishing gradient problem in deep networks
    - Allows training networks 100+ layers deep
    
    Args:
        dropout_rates: Tuple of 3 dropout rates (layer1, layer2, layer3)
                      Controls regularization strength (Section 8.2)
        learning_rate: Initial learning rate for Adam optimizer (Section 4)
        
    Returns:
        Compiled Keras model ready for training
        
    PDF References: Section 2.3 (Architecture), Section 2.7 (Regularization)
    """
    
    # =========================================================
    # INPUT LAYER
    # =========================================================
    # Input shape: (224, 224, 3) required by ResNet50
    # 224x224 pixels, 3 channels (RGB color)
    inputs = Input(shape=(224, 224, 3), name='input_layer')
    
    # =========================================================
    # DATA AUGMENTATION LAYERS (Section 1.5)
    # =========================================================
    # Applied only during training (automatically disabled at test time)
    # Artificially expands training set to prevent overfitting
    
    # Random horizontal flip - creates mirror images
    # Improves invariance to left-right orientation
    x = RandomFlip("horizontal", name='augmentation_flip')(inputs)
    
    # Random rotation ±20% = ±72 degrees
    # Helps model handle fruits at different angles
    x = RandomRotation(0.2, name='augmentation_rotation')(x)
    
    # Random zoom ±20%
    # Simulates fruits at different distances from camera
    x = RandomZoom(0.2, name='augmentation_zoom')(x)
    
    # =========================================================
    # RESNET50 BASE MODEL (Section 2.2)
    # =========================================================
    # Load pre-trained ResNet50 without top classification layer
    base_model = ResNet50(
        include_top=False,          # Exclude original 1000-class classifier
        weights='imagenet',         # Use ImageNet pre-trained weights
        input_tensor=x,             # Connect to our augmented input
        pooling='avg'               # Global average pooling → 2048-dim vector
    )
    
    # CRITICAL: Freeze all ResNet50 layers (Section 2.3)
    # We use it as fixed feature extractor, not fine-tuning
    # This prevents overfitting on our smaller dataset
    base_model.trainable = False
    
    # =========================================================
    # CUSTOM CLASSIFICATION HEAD (Section 2.3)
    # =========================================================
    # Progressive dropout strategy: increase dropout as we approach output
    # This provides stronger regularization for final predictions
    
    x = base_model.output  # Shape: (None, 2048) from ResNet50
    
    # ------- Dense Layer 1: 2048 → 512 -------
    # Batch Normalization: Normalizes activations (Section 2.6)
    # Formula: y = γ((x - μ)/σ) + β
    # Benefits: Faster training, mild regularization, stable gradients
    x = BatchNormalization(name='bn_1')(x)
    
    # Dropout: Randomly zero out neurons during training (Section 2.7)
    # Forces network to learn redundant representations
    # dropout_rates[0] typically 0.2-0.3 (mild regularization)
    x = Dropout(dropout_rates[0], name='dropout_1')(x)
    
    # Dense layer with ReLU activation (Section 2.5)
    # ReLU: f(x) = max(0, x) - computationally efficient, no vanishing gradient
    # 512 neurons provide sufficient capacity for fruit features
    x = Dense(512, activation='relu', name='fc_1')(x)
    
    # ------- Dense Layer 2: 512 → 256 -------
    x = BatchNormalization(name='bn_2')(x)
    x = Dropout(dropout_rates[1], name='dropout_2')(x)  # Medium dropout
    x = Dense(256, activation='relu', name='fc_2')(x)
    
    # ------- Pre-output Layer -------
    x = BatchNormalization(name='bn_3')(x)
    x = Dropout(dropout_rates[2], name='dropout_3')(x)  # Strongest dropout
    
    # ------- Output Layer: 256 → 6 -------
    # Softmax activation for multi-class classification (Section 2.5)
    # Formula: f(x_i) = exp(x_i) / Σ exp(x_j)
    # Outputs probability distribution summing to 1.0
    outputs = Dense(6, activation='softmax', name='output_layer')(x)
    
    # =========================================================
    # CREATE AND COMPILE MODEL
    # =========================================================
    model = Model(inputs=inputs, outputs=outputs, name='ResNet50_FruitClassifier')
    
    # Compile with Adam optimizer and categorical cross-entropy loss
    # (Section 3: Loss Function, Section 4: Optimizer)
    model.compile(
        # Adam: Adaptive learning rate optimizer (Section 4.1)
        # Combines momentum (β1=0.9) and RMSProp (β2=0.999)
        optimizer=Adam(learning_rate=learning_rate),
        
        # Categorical Cross-Entropy: L = -Σ(y_true * log(y_pred))
        # Appropriate for multi-class classification with softmax (Section 3.1)
        loss='categorical_crossentropy',
        
        # Track multiple metrics during training (Section 6)
        metrics=[
            'accuracy',                                    # Overall accuracy
            TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),  # Top-2 accuracy
            Precision(),                                   # Precision per batch
            Recall()                                       # Recall per batch
        ]
    )
    
    return model

# ============================================================
# BUILD MODEL WITH OPTIMAL DROPOUT RATES
# ============================================================
# These rates from Section 8.2 hyperparameter experiments
# (0.2, 0.3, 0.4) achieved best generalization with -0.56% gap
print("=" * 70)
print("BUILDING TRANSFER LEARNING MODEL")
print("=" * 70)
print("Architecture: ResNet50 (frozen) + Custom Classification Head")
print("Dropout rates: (0.2, 0.3, 0.4) - Optimal from Section 8.2")
print("=" * 70)

model = build_resnet50_model_with_dropout(
    dropout_rates=(0.2, 0.3, 0.4),  # Optimal configuration
    learning_rate=0.001              # Standard Adam learning rate
)

# Display model architecture summary
print("\n" + "=" * 70)
print("MODEL ARCHITECTURE SUMMARY")
print("=" * 70)
model.summary()

# Calculate trainable vs non-trainable parameters
trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])

print("\n" + "=" * 70)
print("PARAMETER BREAKDOWN")
print("=" * 70)
print(f"Total parameters:        {model.count_params():,}")
print(f"Trainable parameters:    {trainable_count:,} ({trainable_count/model.count_params()*100:.1f}%)")
print(f"Non-trainable parameters: {non_trainable_count:,} ({non_trainable_count/model.count_params()*100:.1f}%)")
print(f"\n✓ Only training classification head: {trainable_count:,} parameters")
print(f"✓ ResNet50 base frozen: {non_trainable_count:,} parameters")
print("=" * 70)
```

---

## Cell 6: Training Setup and Execution
```python
# ============================================================
# SECTION 6: TRAINING WITH CALLBACKS
# PDF Report Reference: Section 2.8 (Hyperparameters), Section 6 (Results)
# ============================================================

# ============================================================
# CALLBACKS CONFIGURATION
# ============================================================
# Callbacks optimize training and prevent overfitting (Section 2.8)

callbacks_list = [
    # Early Stopping: Prevents overfitting by stopping when no improvement
    # Monitors validation accuracy - stops if no improvement for 5 epochs
    # Restores weights from best epoch (not last epoch)
    EarlyStopping(
        monitor='val_accuracy',      # Watch validation accuracy
        patience=5,                   # Wait 5 epochs before stopping
        restore_best_weights=True,    # Load best model, not final model
        verbose=1,                    # Print when stopping
        mode='max'                    # Maximize accuracy
    ),
    
    # ReduceLROnPlateau: Reduces learning rate when progress plateaus
    # Helps escape local minima and fine-tune weights (Section 4)
    ReduceLROnPlateau(
        monitor='val_loss',          # Watch validation loss
        factor=0.5,                   # Reduce LR by 50% (new_lr = lr * 0.5)
        patience=3,                   # Wait 3 epochs before reducing
        min_lr=1e-7,                 # Don't go below this value
        verbose=1                     # Print when reducing
    ),
    
    # ModelCheckpoint: Saves best model during training
    # Ensures we don't lose best weights if training continues past optimal point
    ModelCheckpoint(
        'best_fruit_model_optimal.h5',  # Filename for saved model
        monitor='val_accuracy',          # Save when validation accuracy improves
        save_best_only=True,             # Only save if better than previous best
        mode='max',                      # Maximize accuracy
        verbose=1                        # Print when saving
    )
]

# ============================================================
# START TRAINING
# ============================================================
print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print("Configuration:")
print(f"  - Model: ResNet50 Transfer Learning")
print(f"  - Epochs: 25 (with early stopping)")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Learning Rate: 0.001 (Adam optimizer)")
print(f"  - Dropout: (0.2, 0.3, 0.4)")
print(f"  - Training samples: ~{num_train_batches * BATCH_SIZE}")
print(f"  - Validation samples: ~{num_val_batches * BATCH_SIZE}")
print("=" * 70)
print("\nCallbacks active:")
print("  ✓ Early Stopping (patience=5)")
print("  ✓ ReduceLROnPlateau (factor=0.5, patience=3)")
print("  ✓ ModelCheckpoint (save best model)")
print("=" * 70)
print("\nExpected training time: ~10-15 minutes with GPU")
print("Starting training...\n")

# Train the model
# fit() returns history object containing all metrics per epoch
history = model.fit(
    train_ds,                   # Training dataset
    validation_data=val_ds,     # Validation dataset for monitoring
    epochs=25,                   # Maximum epochs (early stopping may end sooner)
    callbacks=callbacks_list,    # Apply all our callbacks
    verbose=1                    # Show progress bar with metrics
)

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETED!")
print("=" * 70)

# Print final training metrics (Section 6.1)
final_epoch = len(history.history['accuracy']) - 1
print(f"\nTraining stopped at epoch: {final_epoch + 1}")
print(f"\nFinal Metrics:")
print(f"  Training Accuracy:     {history.history['accuracy'][final_epoch]:.4f} ({history.history['accuracy'][final_epoch]*100:.2f}%)")
print(f"  Validation Accuracy:   {history.history['val_accuracy'][final_epoch]:.4f} ({history.history['val_accuracy'][final_epoch]*100:.2f}%)")
print(f"  Training Loss:         {history.history['loss'][final_epoch]:.4f}")
print(f"  Validation Loss:       {history.history['val_loss'][final_epoch]:.4f}")
print(f"  Training Precision:    {history.history['precision'][final_epoch]:.4f}")
print(f"  Validation Precision:  {history.history['val_precision'][final_epoch]:.4f}")
print(f"  Training Recall:       {history.history['recall'][final_epoch]:.4f}")
print(f"  Validation Recall:     {history.history['val_recall'][final_epoch]:.4f}")

# Calculate overfitting gap (Section 7.2)
train_acc = history.history['accuracy'][final_epoch]
val_acc = history.history['val_accuracy'][final_epoch]
overfitting_gap = (train_acc - val_acc) * 100

print(f"\nGeneralization Analysis:")
print(f"  Overfitting Gap:       {overfitting_gap:+.2f}%")
if abs(overfitting_gap) < 2:
    print(f"  Assessment: ✅ Excellent generalization (< 2% gap)")
elif abs(overfitting_gap) < 5:
    print(f"  Assessment: ✓ Good generalization (< 5% gap)")
else:
    print(f"  Assessment: ⚠ Consider more regularization (> 5% gap)")

print("=" * 70)
```

---

## Cell 7: Training Results Visualization
```python
# ============================================================
# SECTION 6.1: TRAINING RESULTS VISUALIZATION
# PDF Report Reference: Section 6 (Results) - Figure 3
# ============================================================

print("\nGenerating training visualization...")

# Create figure with 4 subplots for comprehensive analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Transfer Learning Training Results - ResNet50 Fruit Classifier', 
             fontsize=16, fontweight='bold', y=0.995)

# Extract epoch numbers for x-axis
epochs_range = range(1, len(history.history['accuracy']) + 1)

# ============================================================
# Subplot 1: Training & Validation Accuracy
# ============================================================
# Accuracy measures correct predictions / total predictions
# Gap between lines indicates overfitting (train > val) or underfitting (val > train)
axes[0, 0].plot(epochs_range, history.history['accuracy'], 'b-', 
                linewidth=2, label='Training Accuracy', marker='o', markersize=4)
axes[0, 0].plot(epochs_range, history.history['val_accuracy'], 'r-', 
                linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
axes[0, 0].set_title('Model Accuracy Over Epochs', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Epoch', fontsize=11)
axes[0, 0].set_ylabel('Accuracy', fontsize=11)
axes[0, 0].legend(loc='lower right', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0.0, 1.0])

# Add annotation for best validation accuracy
best_val_acc_idx = np.argmax(history.history['val_accuracy'])
best_val_acc = history.history['val_accuracy'][best_val_acc_idx]
axes[0, 0].annotate(f'Best: {best_val_acc:.4f}\nEpoch {best_val_acc_idx + 1}',
                    xy=(best_val_acc_idx + 1, best_val_acc),
                    xytext=(best_val_acc_idx + 1, best_val_acc - 0.15),
                    fontsize=9, ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# ============================================================
# Subplot 2: Training & Validation Loss
# ============================================================
# Loss (categorical cross-entropy) measures prediction error
# Lower is better; converging lines indicate good generalization
axes[0, 1].plot(epochs_range, history.history['loss'], 'b-', 
                linewidth=2, label='Training Loss', marker='o', markersize=4)
axes[0, 1].plot(epochs_range, history.history['val_loss'], 'r-', 
                linewidth=2, label='Validation Loss', marker='s', markersize=4)
axes[0, 1].set_title('Model Loss Over Epochs', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Epoch', fontsize=11)
axes[0, 1].set_ylabel('Loss (Categorical Cross-Entropy)', fontsize=11)
axes[0, 1].legend(loc='upper right', fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# ============================================================
# Subplot 3: Precision
# ============================================================
# Precision = True Positives / (True Positives + False Positives)
# "Of all predictions for class X, how many were correct?"
# Important for minimizing false alarms
axes[1, 0].plot(epochs_range, history.history['precision'], 'b-', 
                linewidth=2, label='Training Precision', marker='o', markersize=4)
axes[1, 0].plot(epochs_range, history.history['val_precision'], 'r-', 
                linewidth=2, label='Validation Precision', marker='s', markersize=4)
axes[1, 0].set_title('Model Precision Over Epochs', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Epoch', fontsize=11)
axes[1, 0].set_ylabel('Precision', fontsize=11)
axes[1, 0].legend(loc='lower right', fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0.0, 1.0])

# ============================================================
# Subplot 4: Recall
# ============================================================
# Recall = True Positives / (True Positives + False Negatives)
# "Of all actual class X samples, how many did we find?"
# Important for not missing positive cases
axes[1, 1].plot(epochs_range, history.history['recall'], 'b-', 
                linewidth=2, label='Training Recall', marker='o', markersize=4)
axes[1, 1].plot(epochs_range, history.history['val_recall'], 'r-', 
                linewidth=2, label='Validation Recall', marker='s', markersize=4)
axes[1, 1].set_title('Model Recall Over Epochs', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Epoch', fontsize=11)
axes[1, 1].set_ylabel('Recall', fontsize=11)
axes[1, 1].legend(loc='lower right', fontsize=10)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0.0, 1.0])

plt.tight_layout()
plt.savefig('transfer learning training results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: transfer learning training results.png (Figure 3 in PDF)")
plt.show()

print("\n" + "=" * 70)
print("KEY INSIGHTS FROM TRAINING CURVES:")
print("=" * 70)
print(f"✓ Best validation accuracy: {best_val_acc:.4f} at epoch {best_val_acc_idx + 1}")
print(f"✓ Training stopped at epoch: {len(epochs_range)}")
print(f"✓ Final overfitting gap: {overfitting_gap:+.2f}%")
print("=" * 70)
```

---

## Cell 8: Dropout Variation Experiment
```python
# ============================================================
# SECTION 8.2: HYPERPARAMETER EXPERIMENT - DROPOUT RATES
# PDF Report Reference: Section 8.2 (Impact of Varying Hyperparameters)
# ============================================================
# This experiment tests different dropout configurations to find optimal
# balance between underfitting (too much dropout) and overfitting (too little)

print("=" * 70)
print("EXPERIMENT: DROPOUT RATE VARIATION")
print("=" * 70)
print("Objective: Find optimal dropout configuration for best generalization")
print("Strategy: Test 3 configurations (Baseline, More Aggressive, Less Aggressive)")
print("=" * 70)

# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================
# Each configuration has 3 dropout rates for the 3 dense layers
# Progressive dropout: rates increase as we approach output layer

dropout_experiments = {
    'Baseline': {
        'rates': (0.3, 0.4, 0.5),  # Original configuration (Section 6)
        'description': 'Original dropout rates from baseline model'
    },
    'More Aggressive': {
        'rates': (0.5, 0.6, 0.7),  # Higher dropout = more regularization
        'description': 'Stronger regularization to combat overfitting'
    },
    'Less Aggressive': {
        'rates': (0.2, 0.3, 0.4),  # Lower dropout = less regularization
        'description': 'Gentler regularization to maintain model capacity'
    }
}

# Dictionary to store results from each configuration
results = {}

print(f"\nRunning {len(dropout_experiments)} configurations...")
print("Expected total time: ~30-45 minutes with GPU\n")

# ============================================================
# MAIN EXPERIMENT LOOP
# ============================================================
for config_name, config in dropout_experiments.items():
    print("\n" + "=" * 70)
    print(f"TRAINING: {config_name}")
    print("=" * 70)
    print(f"Dropout rates: {config['rates']}")
    print(f"Description: {config['description']}")
    print("=" * 70)
    
    # Build model with current dropout configuration
    # Uses same architecture as baseline, only dropout rates change
    model_exp = build_resnet50_model_with_dropout(
        dropout_rates=config['rates'],
        learning_rate=0.001  # Keep learning rate constant across experiments
    )
    
    # Setup callbacks specific to this experiment
    # Save each model with unique filename to avoid overwriting
    exp_callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'dropout_exp_{config_name.replace(" ", "_")}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Train model for this configuration
    history_exp = model_exp.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=exp_callbacks,
        verbose=1  # Show progress
    )
    
    # ============================================================
    # EXTRACT AND STORE RESULTS
    # ============================================================
    # Find epoch with best validation accuracy
    best_epoch = np.argmax(history_exp.history['val_accuracy'])
    
    # Store comprehensive metrics for comparison
    results[config_name] = {
        'dropout_rates': config['rates'],
        'history': history_exp.history,
        'best_epoch': best_epoch + 1,  # Human-readable (1-indexed)
        'train_acc': history_exp.history['accuracy'][best_epoch],
        'val_acc': history_exp.history['val_accuracy'][best_epoch],
        'train_loss': history_exp.history['loss'][best_epoch],
        'val_loss': history_exp.history['val_loss'][best_epoch],
        'overfitting_gap': (history_exp.history['accuracy'][best_epoch] - 
                           history_exp.history['val_accuracy'][best_epoch]) * 100
    }
    
    print(f"\n✓ {config_name} Complete!")
    print(f"  Best Epoch: {results[config_name]['best_epoch']}")
    print(f"  Val Accuracy: {results[config_name]['val_acc']:.4f}")
    print(f"  Overfitting Gap: {results[config_name]['overfitting_gap']:+.2f}%")

print("\n" + "=" * 70)
print("✅ ALL EXPERIMENTS COMPLETED!")
print("=" * 70)

# ============================================================
# RESULTS COMPARISON TABLE
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENTAL RESULTS COMPARISON")
print("=" * 70)
print(f"{'Configuration':<20} {'Dropout Rates':<18} {'Train Acc':<11} {'Val Acc':<11} {'Gap':<10} {'Best Epoch'}")
print("-" * 90)

for config_name, result in results.items():
    rates_str = f"{result['dropout_rates']}"
    print(f"{config_name:<20} {rates_str:<18} "
          f"{result['train_acc']*100:>6.2f}%    "
          f"{result['val_acc']*100:>6.2f}%    "
          f"{result['overfitting_gap']:>+6.2f}%   "
          f"{result['best_epoch']:>3}")

print("=" * 70)

# ============================================================
# VISUALIZATION: COMPARISON CHARTS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Dropout Configuration Comparison', fontsize=16, fontweight='bold')

colors = {'Baseline': 'blue', 'More Aggressive': 'red', 'Less Aggressive': 'green'}

# Subplot 1: Validation Accuracy Comparison
for config_name, result in results.items():
    epochs = range(1, len(result['history']['val_accuracy']) + 1)
    axes[0, 0].plot(epochs, result['history']['val_accuracy'], 
                   color=colors[config_name], linewidth=2, 
                   label=f"{config_name}: {result['dropout_rates']}", marker='o', markersize=3)
axes[0, 0].set_title('Validation Accuracy Across Configurations', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Validation Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Overfitting Gap Comparison
config_names = list(results.keys())
gaps = [results[name]['overfitting_gap'] for name in config_names]
bars = axes[0, 1].bar(config_names, gaps, color=[colors[n] for n in config_names], 
                      edgecolor='black', linewidth=1.5)
axes[0, 1].set_title('Overfitting Gap by Configuration', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Gap (%)')
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 1].grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar, gap in zip(bars, gaps):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{gap:+.2f}%', ha='center', 
                   va='bottom' if gap >= 0 else 'top', fontweight='bold')

# Subplot 3: Training Curves for Best Configuration
best_config = max(results.items(), key=lambda x: x[1]['val_acc'])
best_name = best_config[0]
best_history = best_config[1]['history']
epochs = range(1, len(best_history['accuracy']) + 1)
axes[1, 0].plot(epochs, best_history['accuracy'], 'b-', linewidth=2, label='Training', marker='o')
axes[1, 0].plot(epochs, best_history['val_accuracy'], 'r-', linewidth=2, label='Validation', marker='s')
axes[1, 0].set_title(f'Best Configuration: {best_name}', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: Summary Metrics
ax = axes[1, 1]
ax.axis('off')
summary_text = "OPTIMAL CONFIGURATION FOUND:\n\n"
summary_text += f"Configuration: {best_name}\n"
summary_text += f"Dropout Rates: {best_config[1]['dropout_rates']}\n"
summary_text += f"Validation Accuracy: {best_config[1]['val_acc']*100:.2f}%\n"
summary_text += f"Training Accuracy: {best_config[1]['train_acc']*100:.2f}%\n"
summary_text += f"Overfitting Gap: {best_config[1]['overfitting_gap']:+.2f}%\n"
summary_text += f"Best Epoch: {best_config[1]['best_epoch']}\n\n"
summary_text += "ANALYSIS:\n"
if abs(best_config[1]['overfitting_gap']) < 2:
    summary_text += "✅ Excellent generalization (< 2% gap)"
elif abs(best_config[1]['overfitting_gap']) < 5:
    summary_text += "✓ Good generalization (< 5% gap)"
else:
    summary_text += "⚠ Consider more regularization"
ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round', 
        facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('droput configuration comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: droput configuration comparison.png (Figure 4 in PDF)")
plt.show()

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
print(f"Use '{best_name}' configuration with dropout rates {best_config[1]['dropout_rates']}")
print(f"This achieved the best balance: {best_config[1]['val_acc']*100:.2f}% validation accuracy")
print(f"with only {best_config[1]['overfitting_gap']:+.2f}% overfitting gap")
print("=" * 70)
```

---

## Cell 9: Test Set Evaluation
```python
# ============================================================
# SECTION 6.2-6.4: TEST SET EVALUATION
# PDF Report Reference: Section 6.2, 6.3 (Confusion Matrix), 6.4 (Sample Predictions)
# ============================================================
# Final evaluation on held-out test set to assess real-world performance
# This data was NEVER seen during training or validation

print("=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)
print("Loading best model for final evaluation...")
print("=" * 70)

# Load the best model from optimal dropout configuration
# This model was saved during training by ModelCheckpoint callback
best_model = keras.models.load_model('dropout_exp_Less_Aggressive.h5')
print("✓ Model loaded successfully!")

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
print("\nEvaluating on test set...")
test_results = best_model.evaluate(test_ds, verbose=1)

print("\n" + "=" * 70)
print("TEST SET PERFORMANCE (FINAL RESULTS)")
print("=" * 70)
print(f"Test Loss:              {test_results[0]:.4f}")
print(f"Test Accuracy:          {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
print(f"Test Top-2 Accuracy:    {test_results[2]:.4f} ({test_results[2]*100:.2f}%)")
print(f"Test Precision:         {test_results[3]:.4f}")
print(f"Test Recall:            {test_results[4]:.4f}")
print("=" * 70)

# ============================================================
# GENERATE PREDICTIONS FOR DETAILED ANALYSIS
# ============================================================
print("\nGenerating predictions for confusion matrix and per-class analysis...")

# Collect all true labels and predictions
y_true = []  # Ground truth labels
y_pred = []  # Model predictions

# Iterate through test dataset to get predictions
# Use predict() to get probability distributions for each image
for images, labels in test_ds:
    # Get model predictions (shape: [batch_size, 6])
    predictions = best_model.predict(images, verbose=0)
    
    # Convert probabilities to class indices
    # argmax finds the class with highest probability
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels.numpy(), axis=1)
    
    y_pred.extend(predicted_classes)
    y_true.extend(true_classes)

# Convert to numpy arrays for sklearn functions
y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(f"✓ Generated {len(y_true)} predictions")

# ============================================================
# CONFUSION MATRIX VISUALIZATION (Figure 5)
# ============================================================
print("\nGenerating confusion matrix...")

# Compute confusion matrix
# cm[i, j] = number of samples with true label i predicted as label j
cm = confusion_matrix(y_true, y_pred)

# Create visualization
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Number of Predictions'})
plt.title('Confusion Matrix - Test Set Performance', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('True Class', fontsize=12)
plt.tight_layout()
plt.savefig('confusion matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion matrix.png (Figure 5 in PDF)")
plt.show()

# ============================================================
# PER-CLASS PERFORMANCE METRICS (Section 6.2)
# ============================================================
print("\n" + "=" * 70)
print("PER-CLASS PERFORMANCE ANALYSIS")
print("=" * 70)

# Generate detailed classification report
# Precision, Recall, F1-Score for each class
report = classification_report(y_true, y_pred, target_names=class_names, 
                              digits=4, output_dict=True)

# Create formatted table
print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
print("-" * 70)
for class_name in class_names:
    metrics = report[class_name]
    print(f"{class_name:<15} "
          f"{metrics['precision']:>8.4f}    "
          f"{metrics['recall']:>8.4f}    "
          f"{metrics['f1-score']:>8.4f}    "
          f"{int(metrics['support']):>6}")

print("-" * 70)
print(f"{'Macro Avg':<15} "
      f"{report['macro avg']['precision']:>8.4f}    "
      f"{report['macro avg']['recall']:>8.4f}    "
      f"{report['macro avg']['f1-score']:>8.4f}    "
      f"{len(y_true):>6}")
print("=" * 70)

# Identify best and worst performing classes
class_accuracies = {}
for i, class_name in enumerate(class_names):
    # Class accuracy = diagonal element / row sum
    class_accuracies[class_name] = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0

best_class = max(class_accuracies.items(), key=lambda x: x[1])
worst_class = min(class_accuracies.items(), key=lambda x: x[1])

print(f"\n✅ Best performing class:  {best_class[0]} ({best_class[1]*100:.1f}% accuracy)")
print(f"⚠️  Worst performing class: {worst_class[0]} ({worst_class[1]*100:.1f}% accuracy)")
print("=" * 70)

# ============================================================
# SAMPLE PREDICTIONS VISUALIZATION (Figure 6)
# ============================================================
print("\nGenerating sample predictions visualization...")

# Get one batch of test images for visualization
sample_images, sample_labels = next(iter(test_ds))
sample_predictions = best_model.predict(sample_images, verbose=0)

# Create grid of sample predictions
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
fig.suptitle('Sample Test Set Predictions', fontsize=16, fontweight='bold')

for i in range(16):
    ax = axes[i // 4, i % 4]
    
    # Display image
    img = sample_images[i].numpy()
    img = (img * 255).astype("uint8")  # Convert to displayable format
    ax.imshow(img)
    
    # Get true and predicted labels
    true_idx = np.argmax(sample_labels[i].numpy())
    pred_idx = np.argmax(sample_predictions[i])
    true_class = class_names[true_idx]
    pred_class = class_names[pred_idx]
    confidence = sample_predictions[i][pred_idx] * 100
    
    # Color code: green for correct, red for incorrect
    is_correct = (true_idx == pred_idx)
    color = 'green' if is_correct else 'red'
    symbol = '✓' if is_correct else '✗'
    
    # Create title with prediction info
    title = f"{symbol} True: {true_class}\n"
    title += f"Pred: {pred_class} ({confidence:.1f}%)"
    ax.set_title(title, fontsize=10, fontweight='bold', color=color)
    ax.axis('off')

plt.tight_layout()
plt.savefig('sample test set predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sample test set predictions.png (Figure 6 in PDF)")
plt.show()

print("\n" + "=" * 70)
print("✅ TEST SET EVALUATION COMPLETE!")
print("=" * 70)
print("All figures and metrics saved for PDF report.")
print("=" * 70)
```

---

## Cell 10: Memory-Efficient K-Fold Cross-Validation
```python
# ============================================================
# SECTION 5: MEMORY-EFFICIENT 3-FOLD CROSS-VALIDATION
# PDF Report Reference: Section 5 (Cross-Fold Validation)
# ============================================================
# Robust evaluation using stratified K-fold cross-validation
# Memory-efficient implementation: loads images on-the-fly instead of all at once

print("=" * 70)
print("K-FOLD CROSS-VALIDATION (MEMORY-EFFICIENT)")
print("=" * 70)
print("Configuration:")
print("  - K = 3 folds (reduced from 5 to save time/memory)")
print("  - Stratified: Each fold maintains class distribution")
print("  - Memory-efficient: Images loaded on-the-fly during training")
print("=" * 70)

import gc  # Garbage collector for memory management

# ============================================================
# STEP 1: COLLECT FILE PATHS (NOT IMAGES)
# ============================================================
# Instead of loading all images into RAM, we collect file paths
# and load images batch-by-batch during training

print("\nStep 1: Collecting file paths from training directory...")
TRAIN_PATH = '/content/drive/MyDrive/CS4287_Assignment/data/fruits_classification/train'

# Lists to store file paths and corresponding labels
file_paths = []
labels = []

# Walk through training directory to collect all image paths
for class_idx, class_name in enumerate(sorted(os.listdir(TRAIN_PATH))):
    class_path = os.path.join(TRAIN_PATH, class_name)
    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(os.path.join(class_path, img_file))
                labels.append(class_idx)  # Integer label (0-5)

# Convert to numpy arrays for sklearn compatibility
file_paths = np.array(file_paths)
labels = np.array(labels)

print(f"✓ Collected {len(file_paths):,} file paths")
print(f"✓ Memory usage: ~{len(file_paths) * 200 / 1024 / 1024:.1f} MB (paths only, not images!)")
print(f"✓ Classes: {class_names}")

# ============================================================
# HELPER FUNCTION: CREATE DATASET FROM PATHS
# ============================================================
def create_dataset_from_paths(paths, labels, batch_size=32, shuffle=True, augment=False):
    """
    Create TensorFlow dataset that loads images on-the-fly from file paths.
    
    This approach is memory-efficient because:
    - Images are NOT loaded into RAM all at once
    - Each batch loads only 32 images at a time
    - After batch is processed, memory is freed for next batch
    
    Args:
        paths: Array of file paths to images
        labels: Array of integer labels (0-5)
        batch_size: Number of images per batch
        shuffle: Whether to shuffle data
        augment: Whether to apply data augmentation (only for training)
    
    Returns:
        tf.data.Dataset ready for training/validation
    """
    
    def load_and_preprocess_image(path, label):
        """Load single image from disk and preprocess it."""
        # Read image file
        img = tf.io.read_file(path)
        # Decode JPEG/PNG to tensor
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        # Resize to 224x224 (ResNet50 requirement)
        img = tf.image.resize(img, [224, 224])
        # Normalize to [0, 1] range
        img = img / 255.0
        # Convert label to one-hot encoding
        label_onehot = tf.one_hot(label, depth=6)
        return img, label_onehot
    
    # Create dataset from file paths (NOT loading images yet)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    if shuffle:
        # Shuffle with large buffer for better randomization
        dataset = dataset.shuffle(buffer_size=len(paths))
    
    # Map: Load and preprocess images (happens during training, not now)
    dataset = dataset.map(load_and_preprocess_image, 
                         num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch images together
    dataset = dataset.batch(batch_size)
    
    # Prefetch: Prepare next batch while GPU processes current batch
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# ============================================================
# STEP 2: INITIALIZE STRATIFIED K-FOLD
# ============================================================
print("\nStep 2: Setting up 3-fold stratified cross-validation...")

# Stratified K-Fold ensures each fold has same class distribution as full dataset
# Important for imbalanced datasets (we have slight imbalance)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Storage for cross-validation results
cv_results = {
    'fold': [],
    'train_acc': [],
    'val_acc': [],
    'test_acc': [],
    'train_loss': [],
    'val_loss': [],
    'test_loss': [],
    'epochs_trained': [],
    'best_epoch': []
}

fold_histories = []  # Store training history for each fold

# ============================================================
# STEP 3: CROSS-VALIDATION LOOP
# ============================================================
print("\n" + "=" * 70)
print("Starting cross-validation training...")
print("Expected time: ~30-45 minutes with GPU (3 folds × 10-15 min each)")
print("=" * 70)

for fold, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels), 1):
    print("\n" + "=" * 70)
    print(f"FOLD {fold}/3")
    print("=" * 70)
    
    # Split file paths and labels for this fold
    train_paths = file_paths[train_idx]
    train_labels = labels[train_idx]
    val_paths = file_paths[val_idx]
    val_labels = labels[val_idx]
    
    print(f"Training samples: {len(train_paths):,}")
    print(f"Validation samples: {len(val_paths):,}")
    
    # Create datasets (images will be loaded on-the-fly during training)
    train_fold_ds = create_dataset_from_paths(train_paths, train_labels, 
                                              batch_size=32, shuffle=True)
    val_fold_ds = create_dataset_from_paths(val_paths, val_labels, 
                                            batch_size=32, shuffle=False)
    
    # Build fresh model for this fold
    print(f"\nBuilding model for fold {fold}...")
    model_fold = build_resnet50_model_with_dropout(
        dropout_rates=(0.2, 0.3, 0.4),  # Use optimal configuration from Section 8.2
        learning_rate=0.001
    )
    
    # Setup callbacks
    fold_callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, 
                     restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                         patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(f'cv_fold_{fold}.h5', monitor='val_accuracy',
                       save_best_only=True, mode='max', verbose=0)
    ]
    
    # Train model for this fold
    print(f"\nTraining fold {fold}...")
    history_fold = model_fold.fit(
        train_fold_ds,
        validation_data=val_fold_ds,
        epochs=25,
        callbacks=fold_callbacks,
        verbose=1
    )
    
    # Evaluate on validation set
    val_results = model_fold.evaluate(val_fold_ds, verbose=0)
    
    # Evaluate on test set (same test set for all folds)
    test_results = model_fold.evaluate(test_ds, verbose=0)
    
    # Store results
    best_epoch_idx = np.argmax(history_fold.history['val_accuracy'])
    cv_results['fold'].append(fold)
    cv_results['train_acc'].append(history_fold.history['accuracy'][best_epoch_idx])
    cv_results['val_acc'].append(history_fold.history['val_accuracy'][best_epoch_idx])
    cv_results['test_acc'].append(test_results[1])
    cv_results['train_loss'].append(history_fold.history['loss'][best_epoch_idx])
    cv_results['val_loss'].append(history_fold.history['val_loss'][best_epoch_idx])
    cv_results['test_loss'].append(test_results[0])
    cv_results['epochs_trained'].append(len(history_fold.history['accuracy']))
    cv_results['best_epoch'].append(best_epoch_idx + 1)
    
    fold_histories.append(history_fold)
    
    print(f"\n✓ Fold {fold} Complete!")
    print(f"  Val Accuracy:  {cv_results['val_acc'][-1]:.4f} ({cv_results['val_acc'][-1]*100:.2f}%)")
    print(f"  Test Accuracy: {cv_results['test_acc'][-1]:.4f} ({cv_results['test_acc'][-1]*100:.2f}%)")
    print(f"  Best Epoch: {cv_results['best_epoch'][-1]}")
    
    # CRITICAL: Clear memory before next fold
    # Prevents "out of RAM" crashes
    del model_fold, train_fold_ds, val_fold_ds, history_fold
    keras.backend.clear_session()  # Clear Keras session
    gc.collect()  # Force garbage collection
    print(f"  Memory cleared for next fold")

# ============================================================
# STEP 4: AGGREGATE AND ANALYZE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("✅ CROSS-VALIDATION COMPLETE!")
print("=" * 70)

# Convert results to DataFrame for easy analysis
results_df = pd.DataFrame(cv_results)

print("\nPER-FOLD RESULTS:")
print(results_df.to_string(index=False))

# Calculate mean and standard deviation across folds
print("\n" + "=" * 70)
print("CROSS-VALIDATION SUMMARY STATISTICS")
print("=" * 70)
print(f"Validation Accuracy:  {np.mean(cv_results['val_acc'])*100:.2f}% ± {np.std(cv_results['val_acc'])*100:.2f}%")
print(f"Test Accuracy:        {np.mean(cv_results['test_acc'])*100:.2f}% ± {np.std(cv_results['test_acc'])*100:.2f}%")
print(f"Validation Loss:      {np.mean(cv_results['val_loss']):.4f} ± {np.std(cv_results['val_loss']):.4f}")
print(f"Test Loss:            {np.mean(cv_results['test_loss']):.4f} ± {np.std(cv_results['test_loss']):.4f}")
print(f"Average Epochs:       {np.mean(cv_results['epochs_trained']):.1f} ± {np.std(cv_results['epochs_trained']):.1f}")
print("=" * 70)

# ============================================================
# STEP 5: VISUALIZATIONS
# ============================================================
print("\nGenerating cross-validation visualizations...")

# Figure 7: Training curves for all folds
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Training Curves Across 3 Folds', fontsize=14, fontweight='bold')

for fold_idx, history in enumerate(fold_histories, 1):
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    
    # Accuracy
    axes[fold_idx-1].plot(epochs_range, history.history['accuracy'], 
                         'b-', linewidth=2, label='Train Acc', marker='o', markersize=3)
    axes[fold_idx-1].plot(epochs_range, history.history['val_accuracy'], 
                         'r-', linewidth=2, label='Val Acc', marker='s', markersize=3)
    axes[fold_idx-1].set_title(f'Fold {fold_idx}', fontsize=12, fontweight='bold')
    axes[fold_idx-1].set_xlabel('Epoch')
    axes[fold_idx-1].set_ylabel('Accuracy')
    axes[fold_idx-1].legend()
    axes[fold_idx-1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Training curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Training curves.png (Figure 7 in PDF)")
plt.show()

# Figure 8: Box plots of metrics
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('3-Fold Cross-Validation Results Distribution', fontsize=14, fontweight='bold')

metrics = ['val_acc', 'test_acc', 'val_loss', 'test_loss', 'epochs_trained', 'best_epoch']
titles = ['Validation Accuracy', 'Test Accuracy', 'Validation Loss', 
          'Test Loss', 'Epochs Trained', 'Best Epoch']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    row = idx // 3
    col = idx % 3
    axes[row, col].boxplot([cv_results[metric]], labels=[''])
    axes[row, col].scatter([1]*3, cv_results[metric], c='red', s=100, zorder=3)
    axes[row, col].set_title(title, fontsize=11, fontweight='bold')
    mean_val = np.mean(cv_results[metric])
    std_val = np.std(cv_results[metric])
    axes[row, col].text(1.3, mean_val, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                       fontsize=9, verticalalignment='center')
    axes[row, col].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('3 fold cross validation results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 3 fold cross validation results.png (Figure 8 in PDF)")
plt.show()

print("\n" + "=" * 70)
print("✅ CROSS-VALIDATION ANALYSIS COMPLETE!")
print("=" * 70)
print("Key Insights:")
print(f"  ✓ Model generalizes with {np.mean(cv_results['test_acc'])*100:.2f}% test accuracy")
print(f"  ✓ Low variance (±{np.std(cv_results['test_acc'])*100:.2f}%) indicates stable performance")
print(f"  ✓ Consistent across all {len(cv_results['fold'])} folds")
print("=" * 70)
```

---

## 🎯 COMPLETION INSTRUCTIONS

### Copy Order:
1. **Cell 1** (Header) → Add at top of notebook
2. **Cell 2** (Imports) → Replace existing imports
3. **Cell 3** (Data Loading) → After imports
4. **Cell 4** (Visualization) → After data loading
5. **Cell 5** (Model Building) → Before training
6. **Cell 6** (Training) → Main training cell
7. **Cell 7** (Training Viz) → After training
8. **Cell 8** (Dropout Experiment) → In "Hyperparameters" section
9. **Cell 9** (Test Evaluation) → In "Results" section
10. **Cell 10** (Cross-Validation) → In "Validation" section

### Important Notes:
- Student names already updated: Todor Aleksandrov (22336303) and Darragh Kennedy (22346945)
- Update file paths if your Google Drive structure differs
- These comments reference specific PDF sections - ensure section numbers match
- Each cell is self-contained and can run independently (after prerequisites)

---

**✅ CODE COMMENTING COMPLETE!**  
All critical lines are now comprehensively documented with:
- Purpose explanations
- PDF section cross-references
- Mathematical formulas where relevant
- Best practices justifications
