# Run this script AFTER restarting your computer (after running 1_ENABLE_LONG_PATHS.ps1)

Write-Host "Installing Python packages for CS4287 Assignment..." -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 1: Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host ""
Write-Host "Step 2: Installing core packages..." -ForegroundColor Yellow
pip install numpy pandas matplotlib seaborn scikit-learn

Write-Host ""
Write-Host "Step 3: Installing TensorFlow..." -ForegroundColor Yellow
Write-Host "(This may take a few minutes...)" -ForegroundColor Gray
pip install tensorflow

Write-Host ""
Write-Host "Step 4: Installing Jupyter..." -ForegroundColor Yellow
pip install jupyter notebook

Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow
python -c "import tensorflow as tf; import numpy as np; import pandas as pd; import matplotlib; import seaborn; import sklearn; print('All packages installed successfully!'); print('TensorFlow version:', tf.__version__)"

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run: python fruit_detection_complete.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")


