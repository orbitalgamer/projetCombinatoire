$n = Read-Host "Combien de en // l'heuristique de thibaut ?"

if ($n -match '^\d+$') {
    $n = [int]$n

    for ($i = 1; $i -le $n; $i++) {
        Write-Host "exec #$i"
        
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "python 'genetics_thib_v3.py'"
    }
} else {
    Write-Host "Veuillez entrer un nombre valide."
}
