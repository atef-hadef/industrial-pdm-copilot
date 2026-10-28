# Changelog

Toutes les évolutions notables du projet sont consignées ici.

## Non publié

### Ajouté
- `src/data/labeling.py` : `add_rul(df, cap=125)` calcule la RUL du jeu train (cycle max du moteur − cycle courant), plafonnée à `cap` (RUL piecewise-linear) ; `cap=None` donne la RUL linéaire brute. Vérification via `python -m src.data.labeling`.
- `src/data/preprocess.py` : `normalize(train, test)` garde les 14 capteurs utiles (`USEFUL_SENSORS`) et applique un min-max [0, 1] appris sur le train seulement, puis réappliqué au test (pas de fuite de données).
- `src/data/windows.py` : `make_windows(df, window_size=30)` crée des fenêtres glissantes par moteur, X `(n_fenetres, 30, 14)` et y = RUL au dernier cycle de la fenêtre. Vérification via `python -m src.data.windows`.
- `src/models/lstm.py` : `LSTMRegressor`, un LSTM 2 couches (hidden 64, dropout 0,2) suivi d'une couche linéaire, qui prédit la RUL au dernier cycle de la fenêtre.
- `src/models/train.py` : entraînement du LSTM sur FD001 (`python -m src.models.train`).
  - Découpage train/validation **par moteur** (80/20, seed 42), fait avant la normalisation et les fenêtres.
  - MSE, Adam (lr 1e-3), 20 epochs, batch 256, avec la RMSE train/validation affichée à chaque epoch.
  - La cible est divisée par le plafond pendant l'entraînement : avec la RUL brute (0–125), la RMSE validation n'était encore qu'à 47 après 20 epochs.
  - Le meilleur modèle (selon la validation) est sauvegardé dans `models/lstm_fd001.pt`, avec le scaler dans `models/scaler_fd001.joblib`.
- `src/data/windows.py` : `last_windows(df, window_size=30)` renvoie la dernière fenêtre de chaque moteur (cas du jeu test).
- `src/models/evaluate.py` : évaluation sur le test FD001 (`python -m src.models.evaluate`).
  - Le test est normalisé avec le scaler sauvegardé, sans nouveau fit.
  - Le modèle prédit la RUL à partir de la dernière fenêtre de chaque moteur ; la prédiction est remise en cycles (× 125) puis bornée dans [0, 125].
  - Les vraies RUL sont plafonnées à 125.
  - Calcul de la RMSE et du score NASA (asymétrique, pénalise davantage les prédictions tardives).
  - Graphique prédit vs vrai enregistré dans `reports/eval_fd001.png`.
- `src/models/predict.py` : `load_model()` et `predict_rul()`, partagés par l'évaluation et le dashboard, avec le même calcul (dernière fenêtre, × cap, borné dans [0, 125]).
- `src/dashboard/app.py` : dashboard Streamlit « salle de contrôle » (`streamlit run src/dashboard/app.py`).
  - Thème sombre (CSS personnalisé + `.streamlit/config.toml`) et barre latérale de navigation.
  - 4 cartes KPI : critiques (RUL < 30), alerte (30 à 60), RUL moyenne, total de moteurs.
  - Vue Flotte : barres Plotly colorées par risque, avec les seuils, et registre trié par risque avec badges de statut.
  - Vue Moteur : jauge Plotly à zones rouge/orange/vert, et courbes de 6 capteurs où la fenêtre de 30 cycles utilisée par le modèle est grisée.
  - Le modèle est mis en cache (`@st.cache_resource`), ainsi que les prédictions de la flotte (`@st.cache_data`).
- `requirements.txt` : ajout de `plotly` et `streamlit`.

### Modifié
- Dashboard, mise en forme :
  - barre d'en-tête « Supervision prédictive de la flotte », avec une pastille d'état (moteurs · critiques · en alerte) ;
  - jauge graduée de 0 à 125, avec la valeur affichée sobrement dessous, au format « N /125 cycles » ;
  - espacements unifiés : 20 px entre toutes les cartes, même padding et même arrondi partout. Les espaceurs manuels sont supprimés, ainsi que la marge négative que Streamlit ajoute sous les blocs Markdown.
- Préparation du repo public :
  - `README.md` complet : problème, démo, résultats, architecture, stack, lancement, structure, feuille de route ;
  - `assets/` : captures du dashboard (vue flotte, vue moteur) ;
  - cahier des charges retiré du dépôt et de tout l'historique Git (document de travail interne) ;
  - lien de téléchargement du dataset : « NASA Turbofan Jet Engine Data Set » sur Kaggle.
- README de niveau produit :
  - badges et sommaire cliquable ;
  - GIF animé du dashboard en tête (`assets/dashboard-demo.gif`) ;
  - diagramme d'architecture Mermaid ;
  - résultats illustrés par le graphique prédit vs réel (`assets/eval-fd001.png`) ;
  - section « Décisions d'ingénierie clés » ;
  - URL de clonage GitHub.
- `LICENSE` : licence MIT (Atef Ben Hadef).
- `README.md` traduit en anglais, avec la même structure : badges, GIF, diagramme Mermaid, captures, résultats, « Key Engineering Decisions ».
- Dashboard traduit en anglais (textes affichés seulement ; design, couleurs et logique inchangés) :
  - navigation « Fleet View » / « Engine View » ;
  - statuts Critical / Warning / Healthy ;
  - KPI, en-tête, jauge, capteurs, « Fleet registry ».
- Graphique d'évaluation (`evaluate.py`) traduit en anglais.
- Visuels régénérés en anglais et renommés : `dashboard-fleet.png`, `dashboard-engine.png`, `dashboard-demo.gif`, `eval-fd001.png`.

### Corrigé
- Graphique de la flotte : axe X forcé en catégories. Plotly lisait les libellés `#081` comme des nombres et perdait le tri par RUL.

### Constaté
- FD001 train avec `cap=125` : 8 131 lignes sur 20 631 (39,4 %) sont au plafond.
- FD001 train fenêtré : X `(17731, 30, 14)`, y `(17731,)`. Après normalisation, les capteurs du test sortent légèrement de [0, 1] (de −0,04 à 1,08), ce qui est attendu.
- Premier LSTM : meilleure RMSE validation **14,23 cycles** (epoch 20, contre 13,00 en train). Elle est calculée sur toutes les fenêtres des 20 moteurs de validation, donc elle n'est pas encore comparable au benchmark test.
- **Test FD001 (100 moteurs, dernière fenêtre) : RMSE 13,47 cycles, score NASA 318,5.** 56 prédictions sont tardives et 44 précoces ; la plus grosse erreur est −46,2 cycles (moteur 93).

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
