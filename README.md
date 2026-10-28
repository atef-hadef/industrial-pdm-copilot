# Industrial Predictive Maintenance Copilot

> Plateforme end-to-end de maintenance prédictive. Prédiction de durée de vie résiduelle (RUL) par deep learning, industrialisée en pipeline MLOps (MLflow / Docker / FastAPI), surmontée d'un **copilote IA générative (RAG)** qui transforme la prédiction en **décision** : diagnostic expliqué, manuel cité, ordre de travail rédigé.

**Dataset :** NASA C-MAPSS (dégradation de turboréacteurs).

## Pourquoi ce projet

Un seul projet qui couvre trois compétences :

1. **Rigueur data science** — prédiction de RUL (régression sur séries temporelles de dégradation).
2. **Industrialisation (MLOps)** — un modèle déployé, monitoré, versionné.
3. **IA générative appliquée** — un copilote qui va jusqu'à l'**action** (ordre de travail).

## Statut

Palier 0 — mise en place du projet. Voir [CHANGELOG.md](CHANGELOG.md).

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows (Linux/macOS : source venv/bin/activate)
pip install -r requirements.txt
```

Placer les fichiers C-MAPSS (`train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt`) dans `data/raw/`.

## Architecture

_À venir._

## Résultats

_À venir._
