# Changelog

Toutes les évolutions notables du projet sont consignées ici.

## Non publié

### Ajouté
- `src/data/labeling.py` : `add_rul(df, cap=125)` calcule la RUL du jeu train (cycle max du moteur − cycle courant), plafonnée à `cap` (RUL piecewise-linear) ; `cap=None` donne la RUL linéaire brute. Vérification via `python -m src.data.labeling`.
- `src/data/preprocess.py` : `normalize(train, test)` garde les 14 capteurs utiles (`USEFUL_SENSORS`) et applique un min-max [0, 1] appris sur le train seulement, puis réappliqué au test (pas de fuite de données).
- `src/data/windows.py` : `make_windows(df, window_size=30)` crée des fenêtres glissantes par moteur, X `(n_fenetres, 30, 14)` et y = RUL au dernier cycle de la fenêtre. Vérification via `python -m src.data.windows`.

### Constaté
- FD001 train avec `cap=125` : 8 131 lignes sur 20 631 (39,4 %) sont au plafond.
- FD001 train fenêtré : X `(17731, 30, 14)`, y `(17731,)`. Après normalisation, les capteurs du test sortent légèrement de [0, 1] (de −0,04 à 1,08), ce qui est attendu.

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
