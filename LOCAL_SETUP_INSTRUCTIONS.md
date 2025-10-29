# 🖥️ Local Setup Instructions (Windows)

## ⚡ Quick 3-Step Process

### Step 1️⃣: Enable Long Paths (Run as Administrator)

1. **Right-click on PowerShell** → Select **"Run as Administrator"**
2. Navigate to your project folder:
   ```powershell
   cd "C:\Users\todor\OneDrive - University of Limerick\Mine\Github\Neural Computing"
   ```
3. Run:
   ```powershell
   .\1_ENABLE_LONG_PATHS.ps1
   ```
4. **RESTART YOUR COMPUTER** (required!)

---

### Step 2️⃣: Install Packages (After Restart)

1. Open **regular PowerShell** (not admin)
2. Navigate to your project folder:
   ```powershell
   cd "C:\Users\todor\OneDrive - University of Limerick\Mine\Github\Neural Computing"
   ```
3. Run:
   ```powershell
   .\2_INSTALL_PACKAGES.ps1
   ```
4. Wait for installation to complete (5-10 minutes)

---

### Step 3️⃣: Run the Project

```powershell
.\3_RUN_PROJECT.ps1
```

OR simply:

```powershell
python fruit_detection_complete.py
```

---

## 📋 What Each Script Does

| Script | Purpose | Requires Admin? | Requires Restart? |
|--------|---------|-----------------|-------------------|
| `1_ENABLE_LONG_PATHS.ps1` | Enables Windows long file paths | ✅ Yes | ✅ Yes |
| `2_INSTALL_PACKAGES.ps1` | Installs all Python packages | ❌ No | ❌ No |
| `3_RUN_PROJECT.ps1` | Runs your fruit detection code | ❌ No | ❌ No |

---

## ⚠️ If Scripts Won't Run

If you get "cannot be loaded because running scripts is disabled", run this **once** in PowerShell as Admin:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 🔍 Manual Installation (If you prefer)

### Step 1: Enable Long Paths (as Admin)
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Then **restart computer**.

### Step 2: Install Packages (after restart)
```powershell
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter notebook
```

### Step 3: Verify
```powershell
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

### Step 4: Run
```powershell
python fruit_detection_complete.py
```

---

## ✅ What You Should See

When running the project, you'll see:

1. Dataset loading progress
2. Model architecture summary
3. Training progress (epochs, accuracy, loss)
4. Visualizations saved to disk
5. Final results and metrics

Training will take **30-60 minutes** depending on your CPU.

---

## 💡 Tips

- **Close other programs** while training for better performance
- **Don't close the terminal** during training
- **Save your work** before running (computer will be busy)
- Training can be interrupted and resumed from checkpoints

---

## 🆘 Troubleshooting

**Problem:** "Script cannot be loaded"
- **Solution:** Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` as Admin

**Problem:** "TensorFlow installation failed"
- **Solution:** Make sure you've restarted after Step 1

**Problem:** "No module named tensorflow.python"
- **Solution:** TensorFlow installation was incomplete. Run `pip uninstall tensorflow -y` then `pip install tensorflow`

**Problem:** Training is too slow
- **Solution:** This is normal on CPU. You can reduce epochs in the code or use a smaller batch size.

---

## 🎯 You're Ready When...

You can run this without errors:

```powershell
python -c "import tensorflow as tf; import numpy as np; print('Ready to go!')"
```

---

## Next Steps After Setup

1. ✅ Run `python fruit_detection_complete.py`
2. ✅ Open and update `CS4287-Assign2-PLACEHOLDER-PLACEHOLDER.ipynb`
3. ✅ Follow `SUBMISSION_CHECKLIST.md`
4. ✅ Write report using `PDF_REPORT_TEMPLATE.md`

Good luck! 🚀

