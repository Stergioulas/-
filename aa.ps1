# Ορισμός των διαδρομών
$sourceDirectory = "C:\ergasies\TL_Project" # Προσαρμόστε αν χρειάζεται
$destinationZip = "C:\ergasies\TL_Project_archive.zip" # Προσαρμόστε αν χρειάζεται
$tempDir = "C:\ergasies\temp_zip_source"

# Αρχεία και φάκελοι προς συμπερίληψη (σχετικά με το $sourceDirectory)
$itemsToInclude = @(
    "app.py",
    "pipeline_scripts", # Ολόκληρος ο φάκελος
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "README.md",
    "report.tex",
    "progress_report.md",
    "project_plan.md",
    "generate_h5ad_files.py"
)

# Δημιουργία προσωρινού φακέλου
If (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir }
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

# Αντιγραφή των επιθυμητών αρχείων/φακέλων στον προσωρινό φάκελο
foreach ($item in $itemsToInclude) {
    $sourceItemPath = Join-Path -Path $sourceDirectory -ChildPath $item
    $destinationItemPath = Join-Path -Path $tempDir -ChildPath $item
    If (Test-Path $sourceItemPath) {
        Copy-Item -Path $sourceItemPath -Destination $destinationItemPath -Recurse -Force
    } Else {
        Write-Warning "Item not found and not copied: $sourceItemPath"
    }
}

# Συμπίεση του προσωρινού φακέλου
Compress-Archive -Path "$tempDir\*" -DestinationPath $destinationZip -Force

# Διαγραφή του προσωρινού φακέλου
Remove-Item -Recurse -Force $tempDir

Write-Host "ZIP file created at $destinationZip"