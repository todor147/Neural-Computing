# 🎯 Next Step: Test Set Evaluation

## ✅ What You Just Completed

**Excellent work!** You've successfully:
- ✅ Identified optimal dropout configuration (0.2, 0.3, 0.4)
- ✅ Improved validation accuracy from 76.7% → 78.23%
- ✅ Eliminated overfitting (from +5.84% gap to -0.56%)
- ✅ Reduced training time by 50% (11 epochs vs 22)

---

## 🎯 Next Priority: Test Set Evaluation

**Why this is important:** You need to generate the **confusion matrix** and **per-class metrics** on the test set to complete Section 6 of your report.

**Time required:** 30-60 minutes

---

## 📝 Code to Add to Your Colab Notebook

### **Cell 1** - Markdown Header

```markdown
## Test Set Evaluation

Evaluate the best model (Less Aggressive dropout configuration) on the held-out test set to generate:
- Confusion matrix
- Per-class precision, recall, F1-score
- Sample predictions visualization
```

### **Cell 2** - Load Best Model & Evaluate

```python
# Load the best model from dropout experiments
best_model_path = 'best_fruit_model_less_aggressive.h5'

print("Loading best model...")
best_model = tf.keras.models.load_model(best_model_path)
print(f"✓ Loaded model from {best_model_path}")

# Evaluate on test set
print("\n" + "="*70)
print("EVALUATING ON TEST SET")
print("="*70)

test_results = best_model.evaluate(test_dataset, verbose=1)

print("\n" + "="*70)
print("TEST SET RESULTS")
print("="*70)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
print(f"Test Precision: {test_results[3]:.4f}")
print(f"Test Recall: {test_results[4]:.4f}")
print(f"Test Top-2 Accuracy: {test_results[2]:.4f}")
print("="*70)
```

### **Cell 3** - Generate Predictions for Confusion Matrix

```python
# Get predictions on test set
print("\nGenerating predictions on test set...")

y_true = []
y_pred = []

for images, labels in test_dataset:
    # Get predictions
    predictions = best_model.predict(images, verbose=0)
    
    # Convert one-hot to class indices
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(f"✓ Generated {len(y_true)} predictions")
```

### **Cell 4** - Create Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Test Set\nLess Aggressive Dropout (0.2, 0.3, 0.4)', 
          fontsize=14, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save figure
plt.savefig('confusion_matrix_test.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Saved confusion_matrix_test.png")
```

### **Cell 5** - Per-Class Metrics Table

```python
# Generate classification report
print("\n" + "="*70)
print("PER-CLASS METRICS - TEST SET")
print("="*70)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
print("="*70)

# Create a formatted table
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=range(len(class_names))
)

# Create DataFrame for better formatting
import pandas as pd

metrics_df = pd.DataFrame({
    'Class': class_names,
    'Precision': [f"{p:.4f}" for p in precision],
    'Recall': [f"{r:.4f}" for r in recall],
    'F1-Score': [f"{f:.4f}" for f in f1],
    'Support': support
})

# Add weighted average row
weighted_precision = np.average(precision, weights=support)
weighted_recall = np.average(recall, weights=support)
weighted_f1 = np.average(f1, weights=support)

metrics_df = pd.concat([
    metrics_df,
    pd.DataFrame({
        'Class': ['Weighted Avg'],
        'Precision': [f"{weighted_precision:.4f}"],
        'Recall': [f"{weighted_recall:.4f}"],
        'F1-Score': [f"{weighted_f1:.4f}"],
        'Support': [support.sum()]
    })
], ignore_index=True)

print("\n📊 Formatted Metrics Table:")
print(metrics_df.to_string(index=False))

# Save to CSV
metrics_df.to_csv('test_set_metrics.csv', index=False)
print("\n✓ Saved test_set_metrics.csv")
```

### **Cell 6** - Analyze Misclassifications

```python
# Find most confused pairs
print("\n" + "="*70)
print("MOST COMMON MISCLASSIFICATIONS")
print("="*70)

# Get off-diagonal elements (misclassifications)
misclass = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i, j] > 0:
            misclass.append({
                'True': class_names[i],
                'Predicted': class_names[j],
                'Count': cm[i, j],
                'Percentage': f"{(cm[i, j] / cm[i].sum() * 100):.1f}%"
            })

# Sort by count
misclass_df = pd.DataFrame(misclass).sort_values('Count', ascending=False)

print("\nTop 10 Misclassification Patterns:")
print(misclass_df.head(10).to_string(index=False))

# Calculate per-class error rates
print("\n" + "="*70)
print("PER-CLASS ERROR ANALYSIS")
print("="*70)

for idx, class_name in enumerate(class_names):
    total = cm[idx].sum()
    correct = cm[idx, idx]
    errors = total - correct
    error_rate = (errors / total * 100) if total > 0 else 0
    
    print(f"\n{class_name}:")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct} ({(correct/total*100):.1f}%)")
    print(f"  Errors: {errors} ({error_rate:.1f}%)")
    
    if errors > 0:
        # Find most common misclassification for this class
        confused_with = []
        for j in range(len(class_names)):
            if j != idx and cm[idx, j] > 0:
                confused_with.append((class_names[j], cm[idx, j]))
        
        if confused_with:
            confused_with.sort(key=lambda x: x[1], reverse=True)
            print(f"  Most confused with: {confused_with[0][0]} ({confused_with[0][1]} times)")
```

### **Cell 7** - Visualize Sample Predictions

```python
# Visualize some test predictions
print("\nGenerating sample predictions visualization...")

# Get one batch from test set for visualization
for test_images, test_labels in test_dataset.take(1):
    predictions = best_model.predict(test_images, verbose=0)
    
    # Show 16 samples
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Sample Test Set Predictions\nGreen = Correct, Red = Incorrect', 
                 fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i >= len(test_images):
            break
            
        # Display image
        img = test_images[i].numpy().astype("uint8")
        ax.imshow(img)
        
        # Get true and predicted labels
        true_idx = np.argmax(test_labels[i])
        pred_idx = np.argmax(predictions[i])
        true_label = class_names[true_idx]
        pred_label = class_names[pred_idx]
        confidence = predictions[i][pred_idx] * 100
        
        # Color based on correctness
        color = 'green' if true_idx == pred_idx else 'red'
        
        # Title with prediction
        ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                    fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_predictions_sample.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Saved test_predictions_sample.png")
    break
```

### **Cell 8** - Summary Statistics

```python
# Final summary
print("\n" + "="*70)
print("📊 TEST SET EVALUATION SUMMARY")
print("="*70)
print(f"Model: Less Aggressive Dropout (0.2, 0.3, 0.4)")
print(f"Test Samples: {len(y_true)}")
print(f"\nOverall Metrics:")
print(f"  Accuracy: {test_results[1]*100:.2f}%")
print(f"  Precision: {weighted_precision*100:.2f}%")
print(f"  Recall: {weighted_recall*100:.2f}%")
print(f"  F1-Score: {weighted_f1*100:.2f}%")
print(f"  Top-2 Accuracy: {test_results[2]*100:.2f}%")

# Compare to validation set
print(f"\nValidation vs Test Comparison:")
print(f"  Validation Accuracy: 78.23%")
print(f"  Test Accuracy: {test_results[1]*100:.2f}%")
print(f"  Difference: {(78.23 - test_results[1]*100):.2f}%")

print("\n✅ TEST SET EVALUATION COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  1. confusion_matrix_test.png")
print("  2. test_set_metrics.csv")
print("  3. test_predictions_sample.png")
print("\nNext: Download these files and add to your PDF report Section 6!")
print("="*70)
```

---

## 📥 After Running

**Download these files from Colab:**
1. `confusion_matrix_test.png` → Add to Section 6.3 of PDF
2. `test_set_metrics.csv` → Copy table to Section 6.2 of PDF  
3. `test_predictions_sample.png` → Add to Section 6.4 of PDF

---

## 📝 Update Your PDF Report

Add to **Section 6.3 (Confusion Matrix)**:

```markdown
### 6.3 Confusion Matrix

![Test Set Confusion Matrix](confusion_matrix_test.png)

**Figure 5:** Confusion matrix on test set (457 samples). The model shows strong diagonal indicating good overall performance. Most common confusions: [describe based on your results].
```

Add to **Section 6.2 (Final Model Performance)**:

Copy the metrics table from CSV and add analysis of:
- Which classes perform best/worst
- How test performance compares to validation
- Any surprising patterns

---

## ⏱️ Time Breakdown

- **Running code:** 5-10 minutes
- **Analyzing results:** 10-15 minutes  
- **Updating PDF report:** 15-20 minutes
- **Total:** 30-45 minutes

---

## 🎯 After Test Evaluation

Your remaining tasks (in order):

1. ✅ **Test Set Evaluation** ← YOU'LL BE HERE NEXT
2. **Statement of Work** (20 min) - Write team contributions
3. **Generative AI Log** (30 min) - Document AI usage
4. **Code Comments** (1-2 hours) - Comment every critical line
5. **(Optional)** Cross-fold validation if you have extra time

---

**Ready to run the test evaluation? Copy the cells above into your Colab notebook after your dropout experiment!** 🚀

