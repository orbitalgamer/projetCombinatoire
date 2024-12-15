$n = Read-Host "Combien de en // l'heuristique de thibaut ?"

if ($n -match '^\d+$') {
    $n = [int]$n

    scp .\genetics_thib_v3.py cyril@100.87.194.42:/home/cyril/Desktop/projet/ #update le fichir
    scp -r .\data\ cyril@100.87.194.42:/home/cyril/Desktop/projet/ #update le fichir
    scp .\genetics_thib_v3.py root@100.97.119.110:/root/graph/
    scp -r .\data\ root@100.97.119.110:/root/graph/ #update le fichir

    for ($i = 1; $i -le $n; $i++) {
        Write-Host "exec #$i a distance"
        

        # ssh cyril@100.87.194.42 "python3 /home/cyril/Desktop/projet/genetics_thib_v3.py"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "echo 'pc-fixe';ssh cyril@100.87.194.42 'cd ~/Desktop/projet && /home/cyril/anaconda3/bin/python genetics_thib_v3.py'"
        # ssh root@100.97.119.110 "python3 /root/graph/genetics_thib_v3.py"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "echo 'nas'; ssh root@100.97.119.110 'cd ~/graph && python3 genetics_thib_v3.py'"
    }
} else {
    Write-Host "Veuillez entrer un nombre valide."
}
