# Set the path to the dataset folder (relative to where the script is run)
$rootPath = "..\..\PhD\datasets"

# Verify the dataset folder exists before running
if (-Not (Test-Path -Path $rootPath)) {
    Write-Host "Error: Could not find the 'datasets' folder in the current directory." -ForegroundColor Red
    Exit
}

# Get all .vmrk|vhdr files in the subfolders of 'dataset'
# This matches your dataset/[user_folder]/[filename].vmrk|vhdr structure
$sourceFiles = Get-ChildItem -Path $rootPath -Filter *.vhdr -Recurse

foreach ($file in $sourceFiles) {
    Write-Host "Processing: $($file.FullName)" -ForegroundColor Cyan
    
    # Read the content of the file
    $content = Get-Content -Path $file.FullName
    # Use -replace with a regular expression to target the specific underscore

    $newContent = $content -replace '((?:DataFile|MarkerFile)=sub-[A-Z]+)_(\d+)', '$1$2'    
    # Only save the file if changes were actually made
    if ($content -ne $newContent) {
        $newContent | Set-Content -Path $file.FullName
        Write-Host "  -> Successfully updated: $($file.Name)" -ForegroundColor Green
    } else {
        Write-Host "  -> No match found or already updated." -ForegroundColor Gray
    }
}

Write-Host "Operation Complete." -ForegroundColor Yellow