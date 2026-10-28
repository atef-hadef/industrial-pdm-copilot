# Changelog

Toutes les évolutions notables du projet sont consignées ici.

## v0.1 - chargement + EDA (2026-10-28)

### Ajouté
- `src/data/load.py` : `load_cmapss(subset, split)` lit `train_`/`test_<subset>.txt` (26 colonnes nommées, colonnes vides de fin de ligne supprimées) et `load_rul(subset)` lit la RUL réelle du jeu de test. Exécutable via `python -m src.data.load`.
- `notebooks/01_eda.ipynb` : EDA de FD001 (100 moteurs, durée de vie 128–362 cycles, trajectoires de 6 capteurs sur 3 moteurs, détection des capteurs non informatifs).

### Constaté
- 7 capteurs à retirer en FD001 : `sensor_1, 5, 6, 10, 16, 18, 19` (constants ou 2 valeurs) ; 14 capteurs informatifs conservés.

## v0.0 - setup initial (2026-10-28)

### Ajouté
- Arborescence du repo (`src/` et ses sous-packages, `data/processed/`, `notebooks/`, `tests/`, `models/`, `manuals/`).
- Environnement virtuel Python et `requirements.txt` (numpy, pandas, scikit-learn, matplotlib, jupyter, torch).
- `.gitignore`, `README.md` squelette, `CHANGELOG.md`.
