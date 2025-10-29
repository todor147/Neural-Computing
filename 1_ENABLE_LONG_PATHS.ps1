# Run this script as Administrator ONCE, then restart your computer

Write-Host "Enabling Windows Long Paths..." -ForegroundColor Yellow
Write-Host ""

try {
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
                     -Name "LongPathsEnabled" `
                     -Value 1 `
                     -PropertyType DWORD `
                     -Force | Out-Null
    
    Write-Host "SUCCESS!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Long paths have been enabled." -ForegroundColor Green
    Write-Host ""
    Write-Host "IMPORTANT: You MUST restart your computer now!" -ForegroundColor Red
    Write-Host ""
    Write-Host "After restarting, run: 2_INSTALL_PACKAGES.ps1" -ForegroundColor Cyan
} catch {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host ""
    Write-Host "To run as Administrator:" -ForegroundColor Yellow
    Write-Host "1. Right-click on PowerShell" -ForegroundColor Yellow
    Write-Host "2. Select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host "3. Navigate to this folder" -ForegroundColor Yellow
    Write-Host "4. Run: .\1_ENABLE_LONG_PATHS.ps1" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")


