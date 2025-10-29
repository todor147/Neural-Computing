# CS4287 Assignment 2 - Submission Checklist

## Before You Submit

### 1. Notebook Requirements ✓

- [ ] Rename notebook to: `CS4287-Assign2-[ID1]-[ID2].ipynb`
  - Replace [ID1] and [ID2] with actual student IDs
  
- [ ] First cell contains:
  - [ ] Team member names and student IDs
  - [ ] Statement: "Code executes to completion without errors"
  - [ ] Link to third-party source: https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection/data

- [ ] Code Quality:
  - [ ] Every critical line is commented
  - [ ] Comments demonstrate understanding (not just describing)
  - [ ] Code runs from start to finish without errors
  - [ ] All cells execute in order

### 2. PDF Report Requirements ✓

Must include these sections in order:

- [ ] **i. Title Page**
  - Course code, assignment title
  - Team member names and IDs
  - Submission date

- [ ] **ii. Table of Contents**
  - Page numbers for each section

- [ ] **1. The Dataset (2 marks)**
  - [ ] Visualization of key attributes
  - [ ] Data correlation analysis
  - [ ] Preprocessing steps (normalization, resizing)
  - [ ] Class distribution charts
  - [ ] Sample images

- [ ] **2. Network Structure & Hyperparameters (4 marks)**
  - [ ] Architecture diagram
  - [ ] Weight initialization (He normal) explanation
  - [ ] Activation functions (ReLU, Softmax) discussion
  - [ ] Batch normalization explanation
  - [ ] Regularization (Dropout) description
  - [ ] All hyperparameters listed and justified

- [ ] **3. Cost/Loss Function (1 mark)**
  - [ ] Categorical cross-entropy explained
  - [ ] Why this loss function was chosen
  - [ ] Mathematical formula

- [ ] **4. Optimizer (1 mark)**
  - [ ] Adam optimizer explained
  - [ ] Learning rate and parameters justified
  - [ ] Why Adam was chosen over SGD, etc.

- [ ] **5. Cross-Fold Validation (2 marks)**
  - [ ] K-fold methodology explained
  - [ ] Results from each fold
  - [ ] Average performance across folds
  - [ ] Standard deviation

- [ ] **6. Results (2 marks)**
  - [ ] Training/validation accuracy plots
  - [ ] Training/validation loss plots
  - [ ] Final accuracy reported
  - [ ] Precision, recall, F1-score calculated
  - [ ] Confusion matrix included

- [ ] **7. Evaluation (2 marks)**
  - [ ] Analysis of results
  - [ ] Discussion of overfitting/underfitting
  - [ ] Link between results and model choices
  - [ ] Model strengths and weaknesses
  - [ ] Failure case analysis

- [ ] **8. Hyperparameter Impact (3 marks)**
  - [ ] Multiple hyperparameters tested (learning rate, batch size, etc.)
  - [ ] Results compared with charts/tables
  - [ ] Impact analysis
  - [ ] Data augmentation effects if overfitting detected
  - [ ] Final hyperparameter choices justified

- [ ] **9. Statement of Work**
  - [ ] One paragraph per team member
  - [ ] Clear description of individual contributions
  - [ ] Specific tasks assigned to each member

- [ ] **10. Use of Generative AI**
  - [ ] ALL prompts listed
  - [ ] One-line summary of each response
  - [ ] Explanation of how each was used
  - [ ] Declaration of not using AI for English improvement

- [ ] **11. Level of Difficulty (3 marks)**
  - [ ] Paragraph explaining difficulty level
  - [ ] Justification for 2-3 marks:
    - Custom CNN with 4 conv blocks
    - 6-class classification
    - Cross-fold validation
    - Hyperparameter analysis
    - Extensive data augmentation

- [ ] **12. References**
  - [ ] Dataset source cited
  - [ ] TensorFlow/Keras cited
  - [ ] Any papers or tutorials cited
  - [ ] Proper citation format (IEEE/APA)

### 3. Code Quality Checklist ✓

- [ ] **Execution:**
  - Code runs without errors
  - All imports work
  - Dataset path is correct
  - Results are generated

- [ ] **Comments:**
  - Every function is documented
  - Complex logic is explained
  - Comments show understanding, not just description
  - References to PDF sections where applicable

- [ ] **Structure:**
  - Clear section markers
  - Logical flow from data → model → training → evaluation
  - Results are visualized
  - Outputs are clear and labeled

### 4. File Naming ✓

- [ ] Notebook: `CS4287-Assign2-[ID1]-[ID2].ipynb`
- [ ] PDF: `CS4287-Assign2-[ID1]-[ID2].pdf`

### 5. Submission Method ✓

- [ ] Via Sulis Assignment Tool
- [ ] Before 23:59 Saturday, 1st November
- [ ] Both files uploaded (notebook + PDF)

### 6. Final Checks ✓

- [ ] All figures have captions
- [ ] All tables have titles
- [ ] Page numbers on PDF
- [ ] No placeholder text remains
- [ ] No "TODO" markers left
- [ ] Spell-check completed (NOT using AI)
- [ ] Grammar-check completed (NOT using AI)
- [ ] All claims are supported with evidence/results
- [ ] Team member names consistent throughout

### 7. Grading Rubric Self-Assessment ✓

Rate yourself honestly (0-20 scale):

**Code (0-5):**
- [ ] 4-5: Runs to completion, fully commented
- [ ] 2-3: Runs to completion, partially commented
- [ ] 0-1: Does not run or minimal comments

**Report (0-5):**
- [ ] 4-5: Follows spec, depth in all discussions
- [ ] 2-3: Follows spec, depth where necessary
- [ ] 0-1: Doesn't follow spec or lacks depth

**Dataset (0-3):**
- [ ] 3: Rich features, non-linear, extensive visualizations
- [ ] 2: Non-linear, representative, good visualizations
- [ ] 0-1: Linear, noisy, or small with few visualizations

**Technical Implementation (0-7):**
- Network & hyperparameters (0-4)
- Loss function (0-1)
- Optimizer (0-1)
- Cross-fold validation (0-2)

Expected Total: **16-20 (Exemplary)** or **11-15 (Accomplished)**

### 8. Before Final Submission ✓

- [ ] Print the PDF and review it
- [ ] Test the notebook on a fresh Python environment
- [ ] Verify all links work
- [ ] Confirm file sizes are reasonable (<100MB total)
- [ ] Keep a backup copy

---

## Quick Submission Steps:

1. Update team info in notebook header
2. Rename notebook with your IDs
3. Run entire notebook start to finish
4. Save outputs
5. Write PDF report using the results
6. Complete all 12 sections
7. Upload both files to Sulis
8. Verify uploads succeeded

**Good luck!** 🚀
