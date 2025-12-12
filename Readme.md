## Lorsque vous ajoutez de nouvelles bibliothèques, assurez-vous de les ajouter aux libs dans requirements

## 1 Créer d'abord un environnement virtuel
```bash
python -m venv venv
```

## 2 L'activer 
### Windows (PowerShell)
```bash
.\venv\Scripts\Activate.ps1
```

## Le désactiver
```bash
deactivate
```

## Méthode 1 : Utilisation d'un environnement virtuel

1. Assurez-vous d'être à l'intérieur de votre environnement virtuel.
2. Exécutez la commande suivante dans votre terminal :
```bash
pip freeze > requirements.txt
```

## Parfois pip freeze inclut des packages inutiles. Pour générer une liste plus propre

1. Installer pipreqs
```bash
pip install pipreqs
```

2. Exécuter pipreqs dans le dossier de votre projet
```bash
pipreqs .
```

## Maintenant, installer les dépendances depuis requirements.txt
```bash 
pip install -r requirements.txt
```

## Installation de dlib sur Windows

👉 Téléchargez ce fichier :
`dlib-19.24.0-cp310-cp310-win_amd64.whl`
depuis :
https://github.com/cgohlke/wheels/releases

3. Installez-le manuellement

Exemple (remplacez le chemin par votre emplacement de téléchargement) :
```bash
pip install "C:\Users\VotreNom\Downloads\dlib-19.24.0-cp310-cp310-win_amd64.whl"
```

4. Maintenant, installez face-recognition normalement
```bash
pip install face-recognition
```

C'est tout. Pas de CMake, pas de compilateur, pas de drame.