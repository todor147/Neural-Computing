# CS4287 Assignment 2 - Action Plan & Next Steps

## 📊 Current Progress Summary

### What You Have Accomplished ✅

1. **Successfully trained ResNet50 transfer learning model**
   - Validation Accuracy: 76.7%
   - Training completed in 9 epochs (early stopping)
   - Generated comprehensive training plots
   
2. **Dataset prepared and visualized**
   - 6 fruit classes (227K images total)
   - Class distribution analyzed
   - Sample images documented

3. **Report template updated** with your actual results
   - All figures referenced correctly
   - Architecture described (ResNet50 + custom head)
   - Overfitting identified (5.5% gap)

---

## 🔴 CRITICAL TASKS TO COMPLETE (By Priority)

### PRIORITY 1: Hyperparameter Experiments (3 marks) ⚠️ MOST IMPORTANT

**Why Critical:** This section is worth 3 marks AND the assignment specifically requires experiments when overfitting is detected (you have 5.5% gap).

**What to Do:**

#### Experiment 1: Dropout Variation (RECOMMENDED - Quick & Impactful)
```python
# Test different dropout configurations
dropout_configs = [
    (0.3, 0.4, 0.5),  # Current baseline
    (0.5, 0.6, 0.7),  # More aggressive
    (0.2, 0.3, 0.4),  # Less aggressive
]

for d1, d2, d3 in dropout_configs:
    # Rebuild model with new dropout rates
    # Train for 25 epochs
    # Record: train_acc, val_acc, overfitting_gap
```

**Expected Time:** 30-45 minutes per configuration (x3 = ~2 hours)

**Deliverables:**
- Table comparing results
- Plot showing train/val accuracy for each configuration
- Analysis of which configuration reduces overfitting best

#### Experiment 2: Learning Rate Comparison (RECOMMENDED)
```python
learning_rates = [0.01, 0.001, 0.0001]

for lr in learning_rates:
    model = build_model(learning_rate=lr)
    # Train and record convergence speed and final accuracy
```

**Expected Time:** ~2 hours

**Deliverables:**
- Comparison plot of validation accuracy curves
- Table showing: LR, Final Val Acc, Epochs to Converge

#### Experiment 3: Data Augmentation Impact (OPTIONAL BUT GOOD)
```python
# Configuration 1: No augmentation
# Configuration 2: Current augmentation (baseline)
# Configuration 3: More aggressive augmentation

# Compare overfitting gaps
```

**Expected Time:** ~2 hours

---

### PRIORITY 2: Cross-Fold Validation (2 marks) ⚠️ REQUIRED

**Current Status:** ❌ Not implemented

**What to Do:**

```python
from sklearn.model_selection import StratifiedKFold

# 1. Implement K-fold cross-validation (K=5)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Training fold {fold+1}/5")
    # Train model on this fold
    # Record: train_acc, val_acc, test_acc
    results.append({...})

# 2. Calculate mean and std dev across folds
# 3. Create box plot of accuracies
```

**Expected Time:** 3-4 hours (training 5 models)

**Deliverables:**
- Table with results for each fold
- Mean ± Std Dev for all metrics
- Box plot visualization

**Note:** This might be time-consuming. If pressed for time, you could reduce to K=3 folds.

---

### PRIORITY 3: Test Set Evaluation (2 marks - part of Results section)

**Current Status:** ⚠️ Partial (only validation metrics)

**What to Do:**

```python
# 1. Load best model
model.load_weights('best_fruit_model.h5')

# 2. Evaluate on test set
test_loss, test_acc, test_prec, test_rec = model.evaluate(test_dataset)

# 3. Generate confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
y_pred = model.predict(test_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = ... # Get true labels

cm = confusion_matrix(y_true, y_pred_classes)
# Plot heatmap

# 4. Per-class metrics
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# 5. Visualize predictions
# Show grid of images with predicted vs actual labels
```

**Expected Time:** 1 hour

**Deliverables:**
- Confusion matrix heatmap
- Per-class precision/recall/F1 table
- Sample predictions visualization

---

### PRIORITY 4: Documentation & Report Finalization

#### A. Statement of Work (Section 9)
- Write one paragraph per team member
- Describe specific contributions
- **Time:** 30 minutes

#### B. Generative AI Log (Section 10)
- List ALL prompts you used
- One line summary of each response
- One line explaining how you used it
- **Time:** 30 minutes

#### C. Code Comments
**CRITICAL:** Assignment requires "every critical line MUST be commented"

Go through your notebook and add comments explaining:
- What each major block does
- Why you chose specific parameters
- How the architecture works
- Reference back to PDF sections

**Time:** 1-2 hours

#### D. Notebook Metadata
Ensure first three lines are:
```python
# Team Members: [Name 1 - ID1], [Name 2 - ID2]
# Code Status: Runs to completion without errors
# Source: [If you adapted code, link here, otherwise state "Original implementation"]
```

---

## ⏰ TIME ESTIMATE & SCHEDULE

| Task | Priority | Time Required | Can Skip? |
|------|----------|---------------|-----------|
| Dropout Experiments | HIGH | 2-3 hours | ❌ NO |
| Learning Rate Experiments | HIGH | 2 hours | ⚠️ Preferably no |
| Cross-Fold Validation | MEDIUM | 3-4 hours | ⚠️ Worth 2 marks |
| Test Set Evaluation | MEDIUM | 1 hour | ❌ NO |
| Code Comments | HIGH | 1-2 hours | ❌ NO |
| Report Writing | MEDIUM | 2-3 hours | ❌ NO |
| **TOTAL** | | **11-15 hours** | |

**Recommendation:** Focus on at least 2-3 hyperparameter experiments and test set evaluation if time is limited.

---

## 📋 Submission Checklist

### Code (Jupyter Notebook)
- [ ] Named: CS4287-Assign2-ID1-ID2.ipynb
- [ ] First line: Team member names and IDs (comment)
- [ ] Second line: "Code executes to completion without errors" (comment)
- [ ] Third line: Source attribution (comment)
- [ ] Every critical line commented with YOUR explanations
- [ ] Runs from start to finish without errors
- [ ] Generates all plots/figures

### PDF Report
- [ ] Title page with team members and date
- [ ] Table of Contents
- [ ] All 12 sections completed:
  - [x] 1. Dataset (with figures)
  - [x] 2. Network Architecture (ResNet50 described)
  - [x] 3. Loss Function
  - [x] 4. Optimizer
  - [ ] 5. Cross-Fold Validation ⚠️ TODO
  - [x] 6. Results (need test set metrics)
  - [x] 7. Evaluation
  - [ ] 8. Hyperparameter Experiments ⚠️ TODO
  - [ ] 9. Statement of Work
  - [ ] 10. Generative AI Log
  - [x] 11. Level of Difficulty
  - [x] 12. References
- [ ] All figures included and referenced
- [ ] All tables formatted properly
- [ ] Page numbers
- [ ] Proofread (NO AI for English improvement!)

### Before Submission
- [ ] Run notebook top to bottom one final time
- [ ] Verify all plots are generated
- [ ] Check PDF exports correctly
- [ ] Both files ready to upload to Sulis

---

## 💡 Quick Wins to Maximize Marks

1. **Do the dropout experiment** - Directly addresses overfitting (3 marks section)
2. **Generate test set confusion matrix** - Shows thoroughness (2 marks section)
3. **Comment code extensively** - Required for full marks
4. **Run at least one more hyperparameter experiment** - Shows analysis (3 marks)

---

## 🎯 Minimum Viable Submission Strategy

If extremely time-constrained, prioritize:

1. ✅ **Hyperparameter Experiments** (3 marks):
   - Dropout variation (1 experiment)
   - One other (LR or augmentation)
   - Create comparison plots
   
2. ✅ **Test Set Evaluation**:
   - Confusion matrix
   - Per-class metrics table

3. ✅ **Code Comments**:
   - Comment all major sections thoroughly

4. ⚠️ **Cross-Fold Validation**:
   - If no time: Acknowledge limitation in report
   - Say "Time constraints limited to train/val/test split"
   - You'll lose 2 marks but better than incomplete submission

---

## 📞 Questions to Consider

1. **Do you have access to GPU for additional training?** 
   - If yes: Do full experiments
   - If no: Consider reducing epochs or using smaller experiments

2. **How much time before deadline?**
   - 1 week+: Do everything
   - 3-5 days: Skip cross-fold validation if needed
   - <3 days: Minimum viable strategy

3. **Team division:**
   - Person 1: Hyperparameter experiments + code comments
   - Person 2: Test evaluation + report writing

---

## 🚀 Getting Started Right Now

**Immediate Next Steps (Today):**

1. Open your Colab notebook
2. Add a new cell with dropout experiment code (provided above)
3. Start training different dropout configurations
4. While training, work on test set evaluation code
5. Document everything as you go

**Order of Operations:**
```
1. Run dropout experiments (start now - takes time)
2. While models train: Generate test set evaluation
3. Create comparison plots/tables
4. Update PDF report with new results
5. Add code comments
6. Write Statement of Work & AI Log
7. Final proofread
8. Submit!
```

---

## 📈 Expected Grade Impact

Current state (baseline only): **~13-14/20**
- Good baseline implementation
- Missing key experiments
- Missing cross-fold validation

With recommended work: **~17-19/20**
- Complete hyperparameter analysis
- Thorough evaluation
- Well-documented code
- Comprehensive report

---

**Good luck with your submission! Focus on the hyperparameter experiments first - they're worth 3 marks and directly address your overfitting issue.** 🎓

