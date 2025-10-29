# CS4287 Neural Computing - Assignment 2: CNN Fruit Detection

## Project Overview
Convolutional Neural Network implementation for fruit classification using the Kaggle Fruit Detection dataset.

## Team Members
- **Student 1:** [Name] - [Student ID]
- **Student 2:** [Name] - [Student ID]

## Dataset
- **Source:** https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection/data
- **Classes:** 6 (Apple, Banana, Grape, Orange, Pineapple, Watermelon)
- **Total Images:** 8,479
  - Training: 7,108 images
  - Testing: 457 images
  - Validation: 914 images
- **Preprocessing:** Converted from YOLO format to classification format, resized to 224×224

## Files

### For Submission:
1. **CS4287-Assign2-[ID1]-[ID2].ipynb** - Jupyter notebook with complete implementation
2. **Assignment_Report.pdf** - Detailed report covering all required sections

### Supporting Files:
- **fruit_detection_complete.py** - Complete Python implementation
- **requirements.txt** - Required Python packages
- **data/fruits_classification/** - Organized dataset

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Code

**Option A: Python Script**
```bash
python fruit_detection_complete.py
```

**Option B: Jupyter Notebook**
```bash
jupyter notebook CS4287-Assign2-PLACEHOLDER-PLACEHOLDER.ipynb
```

## Model Architecture
- 4 Convolutional blocks (32 → 64 → 128 → 256 filters)
- Batch Normalization after each convolution
- MaxPooling for dimensionality reduction
- Dropout for regularization (0.25 in conv layers, 0.5 in FC layers)
- 2 Fully connected layers (512 → 256 neurons)
- Softmax output (6 classes)

## Training Configuration
- **Optimizer:** Adam (learning rate: 0.001)
- **Loss:** Categorical Cross-Entropy
- **Batch Size:** 32
- **Image Size:** 224×224×3
- **Callbacks:** ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

## Assignment Requirements Checklist

- [x] Dataset loaded and visualized
- [x] Data preprocessing and normalization
- [x] CNN architecture implemented
- [x] Cross-fold validation
- [x] Hyperparameter analysis
- [x] Results visualization (accuracy, loss plots)
- [x] Confusion matrix and classification report
- [x] Extensive code comments
- [ ] PDF report written
- [ ] Notebook renamed with student IDs
- [ ] Team member contributions documented
- [ ] Generative AI usage documented

## Report Sections (As Per Assignment Spec)
1. Title Page
2. Table of Contents
3. The Dataset (visualizations, preprocessing)
4. Network Structure and Hyperparameters
5. Cost/Loss Function
6. Optimizer
7. Cross-Fold Validation
8. Results (plots, metrics)
9. Evaluation
10. Hyperparameter Impact Analysis
11. Statement of Work
12. Use of Generative AI
13. Level of Difficulty
14. References

## Level of Difficulty
This project targets **2-3 marks** for difficulty:
- Custom CNN architecture (4 convolutional blocks)
- Comprehensive data augmentation
- Cross-fold validation implementation
- Hyperparameter analysis
- Multi-class classification (6 classes)

## Third-Party Sources
- Dataset: https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection/data
- TensorFlow/Keras: https://www.tensorflow.org/
- [Add any other sources you referenced]

## Notes
- Code executes to completion without errors
- All critical lines are extensively commented
- Dataset is non-linear with rich features
- Results include precision, recall, and F1-score

## Submission
- **Deadline:** 23:59 Saturday, 1st November (Week 8)
- **Method:** Sulis Assignment Tool
- **Files:** Jupyter Notebook + PDF Report