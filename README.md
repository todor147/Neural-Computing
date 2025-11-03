# CS4287 Neural Computing - Assignment 1: CNN Fruit Classification

## 🎓 Team Information
- **Course:** CS4287 Neural Computing - Semester 1, AY 25/26
- **Assignment:** Convolutional Neural Networks (CNNs)
- **Due Date:** Saturday, November 1st, 2025 at 23:59
- **Weight:** 20% of total module marks

**Team Members:**
- Student 1: [Name] - [ID]
- Student 2: [Name] - [ID]

---

## 📋 Project Overview

This project implements a **Convolutional Neural Network (CNN)** for fruit classification using the Kaggle Fruit Detection dataset. The model classifies images into 6 fruit categories:
- 🍎 Apple
- 🍌 Banana
- 🍇 Grape
- 🍊 Orange
- 🍍 Pineapple
- 🍉 Watermelon

---

## 🚀 Getting Started with Google Colab

**We recommend using Google Colab for FREE GPU access!**

### Quick Steps:
1. Read **`GOOGLE_COLAB_INSTRUCTIONS.md`** for detailed setup
2. Upload dataset to Google Drive
3. Open notebook in Google Colab
4. Enable GPU (Runtime → Change runtime type → GPU)
5. Mount Google Drive and update data paths
6. Run all cells!

See **`GOOGLE_COLAB_INSTRUCTIONS.md`** for complete step-by-step instructions.

---

## 📁 Project Structure

```
Neural Computing/
├── CS4287-Assign2-[ID1]-[ID2].ipynb    # Main Jupyter notebook
├── data/
│   └── fruits_classification/           # Dataset (upload to Google Drive)
│       ├── train/                       # Training images
│       ├── test/                        # Test images
│       └── valid/                       # Validation images
├── GOOGLE_COLAB_INSTRUCTIONS.md         # 👈 START HERE
├── PDF_REPORT_TEMPLATE.md               # Template for your report
├── SUBMISSION_CHECKLIST.md              # Pre-submission checklist
└── README.md                            # This file
```

---

## 🎯 Assignment Requirements

### Deliverables:
1. **PDF Report** including:
   - Dataset analysis and visualization
   - Network architecture diagram
   - Hyperparameters and justification
   - Loss function and optimizer discussion
   - Cross-fold validation
   - Results with plots (accuracy, precision, recall)
   - Hyperparameter experiments
   - Statement of work
   - Generative AI usage log
   - References

2. **Jupyter Notebook** with:
   - Filename: `CS4287-Assign2-[ID1]-[ID2].ipynb`
   - Team member names and IDs in first cell
   - Execution status comment
   - Third-party source links
   - **Every critical line commented** to show understanding

---

## 🏆 Grading - Level of Difficulty

Your project aims for **Level 2-3**:
- Using a custom CNN architecture (similar complexity to VGG/ResNet blocks)
- 6-class fruit classification
- Batch normalization, dropout, He initialization
- Data augmentation
- Cross-fold validation
- Hyperparameter tuning

**Target Grade:** Accomplished (11-15) to Exemplary (16-20)

---

## 📊 Dataset

**Source:** Kaggle - [Fruit Detection Dataset](https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection)

**Statistics:**
- 6 fruit classes
- ~3000+ training images
- ~600+ validation images
- ~600+ test images
- Image size: 224x224 RGB

**Download and Setup:**
1. Download from Kaggle (requires free account)
2. Extract ZIP file
3. Upload `data/fruits_classification/` to Google Drive at: `/MyDrive/CS4287_Assignment/data/`

---

## 🧠 Model Architecture

**Custom CNN** inspired by VGG and modern best practices:
- **4 Convolutional Blocks** (32 → 64 → 128 → 256 filters)
- **Batch Normalization** after each conv layer
- **Max Pooling** for dimensionality reduction
- **Dropout** for regularization (0.25 in conv, 0.5 in FC)
- **He Initialization** for weights
- **ReLU** activation functions
- **Softmax** output for 6-class classification

**Total Parameters:** ~3-5 million (trainable)

---

## 🔧 Hyperparameters

- **Optimizer:** Adam (learning rate: 0.001)
- **Loss Function:** Categorical Cross-Entropy
- **Batch Size:** 32
- **Image Size:** 224x224
- **Epochs:** 50 (with early stopping)
- **Validation Split:** Pre-split (train/valid/test)

**Experiments:** Vary learning rate, batch size, dropout rate, and data augmentation parameters.

---

## 📈 Expected Results

With GPU training on Colab:
- **Training Time:** ~30-90 minutes
- **Expected Accuracy:** 85-95% (validation)
- **Training plots:** Loss and accuracy curves
- **Evaluation metrics:** Confusion matrix, precision, recall, F1-score

---

## ✅ Submission Checklist

Before submitting, check **`SUBMISSION_CHECKLIST.md`**:
- [ ] Student IDs in filename
- [ ] Names and IDs in notebook header
- [ ] Code executes without errors
- [ ] All code commented
- [ ] PDF report follows specification
- [ ] Generative AI usage documented
- [ ] References included
- [ ] Statement of work included

---

## 📚 Key References

1. **He et al. (2015)** - He Initialization: "Delving Deep into Rectifiers"
2. **Ioffe & Szegedy (2015)** - Batch Normalization
3. **Srivastava et al. (2014)** - Dropout
4. **Goodfellow et al. (2016)** - Deep Learning book
5. **TensorFlow/Keras Documentation**

---

## 🤝 Team Collaboration

**Divide work effectively:**
- Member 1: Data preprocessing, visualization, model implementation
- Member 2: Training, evaluation, hyperparameter experiments, report writing
- Both: Review, testing, final submission

Document individual contributions in "Statement of Work" section of report.

---

## ⚠️ Important Notes

1. **Plagiarism:** Cite all sources. Clearly mark reused code vs. your work.
2. **Generative AI:** Log ALL prompts used. Don't use for grammar/style in report.
3. **Interview:** Be prepared to walk through your code in Weeks 13-15.
4. **Code Must Run:** Second line of notebook must state: "Code executes to completion without errors"

---

## 🆘 Need Help?

1. **Read:** `GOOGLE_COLAB_INSTRUCTIONS.md` for setup issues
2. **Check:** `SUBMISSION_CHECKLIST.md` before submitting
3. **Review:** Assignment specification PDF
4. **Contact:** Lecturer with subject "CS4287 Team" for team issues

---

## 📝 Report Template

Use **`PDF_REPORT_TEMPLATE.md`** as a starting point for your PDF report. It includes all required sections with guidance on what to include.

---

## 🎓 Academic Integrity

- Understand every line of code you submit
- Cite all sources properly (see "Cite it Right" - UL Library)
- Document Generative AI use transparently
- Add value beyond sourced material
- Be prepared for code walkthrough interview

**Failure to comply = F grade**

---

## 🏁 Final Steps

1. ✅ Complete training on Google Colab
2. ✅ Export all results (plots, metrics, model)
3. ✅ Write PDF report using template
4. ✅ Review checklist
5. ✅ Submit via Sulis:
   - PDF report
   - Jupyter notebook (.ipynb file)

**Submission Deadline: Saturday, November 1st, 2025 @ 23:59**

---

Good luck with your assignment! 🚀🍎🍌🍇

---

## License
This is an academic project for CS4287 at University of Limerick.
