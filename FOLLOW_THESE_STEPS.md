# 🔧 FIX: Enable Windows Long Paths

## ⚠️ YOU MUST DO THIS TO INSTALL TENSORFLOW

### Step 1: Open PowerShell as Administrator

1. Press `Win + X` on your keyboard
2. Click **"Terminal (Admin)"** or **"Windows PowerShell (Admin)"**
3. Click **"Yes"** on the security prompt

### Step 2: Run This Command

Copy and paste this EXACTLY:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Press Enter.

### Step 3: Restart Your Computer

**You MUST restart** - not just close programs, but full restart!

### Step 4: After Restart, Install TensorFlow

Open a regular PowerShell (not admin) and run:

```powershell
cd "C:\Users\todor\OneDrive - University of Limerick\Mine\Github\Neural Computing"
pip install tensorflow --no-cache-dir
```

This time it will work!

### Step 5: Verify It Works

```powershell
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

You should see: `TensorFlow version: 2.20.0`

### Step 6: Run Your Project

```powershell
python fruit_detection_complete.py
```

---

## 🎯 Summary (Do These In Order)

1. ✅ Open PowerShell as **Admin**
2. ✅ Run the `New-ItemProperty` command
3. ✅ **Restart computer**
4. ✅ Install TensorFlow
5. ✅ Run your project

---

## ❓ Why Do I Need This?

Windows has a 260-character limit on file paths. TensorFlow has some files with very long names that exceed this limit. Enabling "Long Path Support" removes this limitation.

---

## 🆘 Still Having Issues?

If the admin command doesn't work or you don't have admin access, you have two options:

1. **Ask your IT admin** to enable long paths for you
2. **Use Google Colab** - runs in the browser, no installation needed

---

**Start with Step 1 now!** 🚀


