# Google Colab Setup Instructions for CS4287 Assignment

## Overview
This guide will help you run your CNN fruit classification project on Google Colab with **FREE GPU access**.

---

## 🚀 Quick Start (5 Simple Steps)

### Step 1: Upload Your Dataset to Google Drive

1. **Download the dataset from Kaggle:**
   - Go to: https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection
   - Click "Download" (you'll need a Kaggle account - free to create)
   - Extract the ZIP file

2. **Upload to Google Drive:**
   - Open Google Drive (drive.google.com)
   - Create a folder: `CS4287_Assignment`
   - Inside it, upload your entire `data` folder with the fruits_classification structure

Your Google Drive structure should look like:
```
My Drive/
└── CS4287_Assignment/
    └── data/
        └── fruits_classification/
            ├── train/
            │   ├── Apple/
            │   ├── Banana/
            │   ├── Grape/
            │   ├── Orange/
            │   ├── Pineapple/
            │   └── Watermelon/
            ├── test/
            │   └── [same structure]
            └── valid/
                └── [same structure]
```

---

### Step 2: Upload Your Notebook to Google Colab

1. Go to https://colab.research.google.com/
2. Click "File" → "Upload notebook"
3. Upload your `CS4287-Assign2-PLACEHOLDER-PLACEHOLDER.ipynb` file
4. **Important:** Rename the file with your actual student IDs:
   - File → Rename → `CS4287-Assign2-[ID1]-[ID2].ipynb`

---

### Step 3: Enable GPU in Colab

1. Click "Runtime" in the top menu
2. Select "Change runtime type"
3. Under "Hardware accelerator", select **"T4 GPU"** or **"GPU"**
4. Click "Save"

**Why GPU?** Training CNNs on CPU can take hours; with GPU it takes minutes!

---

### Step 4: Mount Google Drive and Update Data Path

Add this cell at the beginning of your notebook (after imports):

```python
# Mount Google Drive to access your dataset
from google.colab import drive
drive.mount('/content/drive')

# Update dataset path to point to your Google Drive
DATASET_PATH = "/content/drive/MyDrive/CS4287_Assignment/data/fruits_classification"
```

**Find the cell in your notebook** that has:
```python
DATASET_PATH = "data/fruits_classification"
```

And change it to:
```python
DATASET_PATH = "/content/drive/MyDrive/CS4287_Assignment/data/fruits_classification"
```

---

### Step 5: Run Your Notebook

1. Click "Runtime" → "Run all" (or press Ctrl+F9)
2. **First time only:** You'll be asked to authorize Google Drive access
   - Click the link
   - Sign in with your Google account
   - Copy the authorization code
   - Paste it back in Colab
3. Your code will now run with GPU acceleration! 🚀

---

## ⚙️ Important Notes

### Free Colab Limitations
- **Session timeout:** 12 hours maximum
- **Idle timeout:** 90 minutes if no code is running
- **GPU allocation:** Not always guaranteed (but usually available)

**Solution:** If you lose connection, just run all cells again. Your Drive data is safe!

### Saving Your Work
- **Notebook autosaves** to your Google Drive if you opened it from Drive
- **To download results:**
  ```python
  # Save your trained model
  model.save('/content/drive/MyDrive/CS4287_Assignment/fruit_model.h5')
  
  # Save plots
  plt.savefig('/content/drive/MyDrive/CS4287_Assignment/results_plot.png')
  ```

### Checking GPU Status
Add this cell to verify GPU is working:
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("TensorFlow version:", tf.__version__)
```

You should see something like: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

---

## 📊 Expected Performance

With GPU enabled:
- **Training time:** ~5-15 minutes per epoch (depends on dataset size)
- **Total training:** ~30-90 minutes for full training
- **Without GPU:** Would take 5-10x longer!

---

## 🐛 Troubleshooting

### Problem: "No module named 'X'"
**Solution:** Install missing package:
```python
!pip install package-name
```

### Problem: "Cannot find dataset path"
**Solution:** Check your Google Drive path:
```python
# List files to verify path
!ls /content/drive/MyDrive/CS4287_Assignment/data/
```

### Problem: "Out of memory"
**Solution:** Reduce batch size:
```python
BATCH_SIZE = 16  # Instead of 32
```

### Problem: "GPU not available"
**Solution:** 
1. Check Runtime → Change runtime type → Ensure GPU is selected
2. Try reconnecting: Runtime → Disconnect and delete runtime → Reconnect
3. Free tier sometimes runs out - try again in a few hours

### Problem: Drive mount fails
**Solution:**
```python
# Force unmount and remount
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)
```

---

## 🎯 Pro Tips

1. **Keep your browser tab active** - Colab may timeout if tab is closed
2. **Use GPU wisely** - Free tier has usage limits; don't waste it on data exploration
3. **Save frequently** - Save model checkpoints to Drive during training
4. **Download results** - Download your trained model and plots before session ends

---

## 📝 For Your Report

When documenting in your PDF report:
- Mention you used **Google Colab with GPU acceleration**
- Include GPU specs: "Tesla T4 GPU with 15GB VRAM" (or whatever you get)
- Note training time with GPU
- This demonstrates efficient use of cloud computing resources!

---

## 📦 Alternative: Upload Dataset Directly to Colab

If Google Drive is slow, you can upload directly to Colab session:

```python
# Upload zip file
from google.colab import files
uploaded = files.upload()  # Select your data.zip

# Extract
!unzip -q data.zip -d /content/
DATASET_PATH = "/content/fruits_classification"
```

**Note:** This data is temporary and will be lost when session ends!

---

## ✅ Checklist Before Running

- [ ] Dataset uploaded to Google Drive
- [ ] Notebook uploaded to Colab
- [ ] GPU enabled in Runtime settings
- [ ] Google Drive mounted in notebook
- [ ] Dataset path updated to Drive path
- [ ] Student IDs added to notebook filename and header
- [ ] Ready to run!

---

## Need Help?

- **Colab Documentation:** https://colab.research.google.com/notebooks/intro.ipynb
- **TensorFlow on Colab:** https://www.tensorflow.org/tutorials

Good luck with your assignment! 🍎🍌🍇🍊🍍🍉

