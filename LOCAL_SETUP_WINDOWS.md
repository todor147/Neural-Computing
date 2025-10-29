# 🖥️ Local Setup for Windows

## Step 1: Enable Windows Long Paths (Required for TensorFlow)

You need to run this **once** as Administrator:

1. **Open PowerShell as Administrator:**
   - Press `Win + X`
   - Click "Windows PowerShell (Admin)" or "Terminal (Admin)"

2. **Run this command:**
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

3. **Restart your computer** (required for changes to take effect)

---

## Step 2: Install Python Packages

After restarting, open a **regular** PowerShell (not admin) in your project folder and run:

```powershell
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter notebook
```

---

## Step 3: Verify Installation

```powershell
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

If this prints a version number, you're ready to go!

---

## Step 4: Run the Code

```powershell
python fruit_detection_complete.py
```

---

## Alternative: Use CPU-only TensorFlow (Faster Install)

If the full TensorFlow installation is too large:

```powershell
pip install tensorflow-cpu
```

This is smaller and works fine for this assignment (training will be a bit slower but totally usable).

---

## Troubleshooting

**Problem:** "tensorflow-cpu is not a valid package"
- **Solution:** Use `pip install tensorflow` instead

**Problem:** Still getting long path errors after restart
- **Solution:** Make sure you restarted the computer, not just PowerShell

**Problem:** Python not found
- **Solution:** Install Python from Microsoft Store or python.org

---

## Quick Commands Summary

```powershell
# 1. Enable long paths (as Admin, then restart)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# 2. Install packages (after restart)
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter notebook

# 3. Run the code
python fruit_detection_complete.py
```

That's it! 🚀


