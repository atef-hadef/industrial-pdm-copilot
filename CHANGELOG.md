# Changelog

All notable changes to this project are documented here.

## Unreleased

### Added
- `src/data/labeling.py`: `add_rul(df, cap=125)` computes the RUL of the train set (engine's max cycle − current cycle), capped at `cap` (piecewise-linear RUL); `cap=None` gives the raw linear RUL. Check with `python -m src.data.labeling`.
- `src/data/preprocess.py`: `normalize(train, test)` keeps the 14 informative sensors (`USEFUL_SENSORS`) and applies a [0, 1] min-max scaling fitted on the train set only, then re-applied to the test set (no data leakage).
- `src/data/windows.py`: `make_windows(df, window_size=30)` builds per-engine sliding windows, X `(n_windows, 30, 14)` and y = RUL at the last cycle of the window. Check with `python -m src.data.windows`.
- `src/models/lstm.py`: `LSTMRegressor`, a 2-layer LSTM (hidden 64, dropout 0.2) followed by a linear layer, which predicts the RUL at the last cycle of the window.
- `src/models/train.py`: LSTM training on FD001 (`python -m src.models.train`).
  - Train/validation split **by engine** (80/20, seed 42), done before normalization and windowing.
  - MSE, Adam (lr 1e-3), 20 epochs, batch size 256, with train/validation RMSE printed at each epoch.
  - The target is divided by the cap during training: with the raw RUL (0–125), validation RMSE was still at 47 after 20 epochs.
  - The best model (according to validation) is saved to `models/lstm_fd001.pt`, with the scaler in `models/scaler_fd001.joblib`.
- `src/data/windows.py`: `last_windows(df, window_size=30)` returns the last window of each engine (test set case).
- `src/models/evaluate.py`: evaluation on the FD001 test set (`python -m src.models.evaluate`).
  - The test set is normalized with the saved scaler, with no new fit.
  - The model predicts the RUL from each engine's last window; the prediction is converted back to cycles (× 125) and clipped to [0, 125].
  - True RULs are capped at 125.
  - RMSE and NASA score (asymmetric, penalizes late predictions more).
  - Predicted-vs-true plot saved to `reports/eval_fd001.png`.
- `src/models/predict.py`: `load_model()` and `predict_rul()`, shared by the evaluation and the dashboard, with the same computation (last window, × cap, clipped to [0, 125]).
- `src/dashboard/app.py`: "control room" Streamlit dashboard (`streamlit run src/dashboard/app.py`).
  - Dark theme (custom CSS + `.streamlit/config.toml`) and sidebar navigation.
  - 4 KPI cards: critical (RUL < 30), warning (30 to 60), mean RUL, total engines.
  - Fleet view: Plotly bars colored by risk, with thresholds, and a registry sorted by risk with status badges.
  - Engine view: Plotly gauge with red/orange/green zones, and 6 sensor curves where the 30-cycle window used by the model is shaded.
  - The model is cached (`@st.cache_resource`), as are the fleet predictions (`@st.cache_data`).
- `requirements.txt`: added `plotly` and `streamlit`.

### Changed
- Dashboard layout:
  - header bar "Predictive fleet monitoring", with a status pill (engines · critical · warning);
  - gauge scaled from 0 to 125, with the value shown discreetly below it, as "N /125 cycles";
  - unified spacing: 20 px between all cards, same padding and corner radius everywhere. Manual spacers are removed, as is the negative margin Streamlit adds under Markdown blocks.
- Public repo preparation:
  - full `README.md`: problem, demo, results, architecture, stack, getting started, structure, roadmap;
  - `assets/`: dashboard screenshots (fleet view, engine view);
  - project specification removed from the repo and from the entire Git history (internal working document);
  - dataset download link: "NASA Turbofan Jet Engine Data Set" on Kaggle.
- Product-grade README:
  - badges and clickable table of contents;
  - animated dashboard GIF at the top (`assets/dashboard-demo.gif`);
  - Mermaid architecture diagram;
  - results illustrated with the predicted-vs-true plot (`assets/eval-fd001.png`);
  - "Key Engineering Decisions" section;
  - GitHub clone URL.
- `LICENSE`: MIT license (Atef Ben Hadef).
- `README.md` translated into English, with the same structure: badges, GIF, Mermaid diagram, screenshots, results, "Key Engineering Decisions".
- Dashboard translated into English (displayed text only; design, colors and logic unchanged):
  - "Fleet View" / "Engine View" navigation;
  - Critical / Warning / Healthy statuses;
  - KPIs, header, gauge, sensors, "Fleet registry".
- Evaluation plot (`evaluate.py`) translated into English.
- Visuals regenerated in English and renamed: `dashboard-fleet.png`, `dashboard-engine.png`, `dashboard-demo.gif`, `eval-fd001.png`.
- Repo fully in English:
  - comments, docstrings, console output and error messages in `src/`;
  - `notebooks/01_eda.ipynb` (markdown, comments, plot labels; re-executed);
  - this changelog.

  Code logic and results are unchanged.

### Fixed
- Fleet chart: X axis forced to categorical. Plotly was parsing the `#081` labels as numbers and losing the RUL ordering.

### Observed
- FD001 train with `cap=125`: 8,131 rows out of 20,631 (39.4%) are at the cap.
- Windowed FD001 train: X `(17731, 30, 14)`, y `(17731,)`. After normalization, test sensors fall slightly outside [0, 1] (−0.04 to 1.08), which is expected.
- First LSTM: best validation RMSE **14.23 cycles** (epoch 20, vs. 13.00 on train). It is computed over all windows of the 20 validation engines, so it is not yet comparable to the test benchmark.
- **FD001 test set (100 engines, last window): RMSE 13.47 cycles, NASA score 318.5.** 56 predictions are late and 44 early; the largest error is −46.2 cycles (engine 93).

## v0.1 - loading + EDA (2026-10-28)

### Added
- `src/data/load.py`: `load_cmapss(subset, split)` reads `train_`/`test_<subset>.txt` (26 named columns, trailing empty columns dropped) and `load_rul(subset)` reads the true RUL of the test set. Runnable with `python -m src.data.load`.
- `notebooks/01_eda.ipynb`: FD001 EDA (100 engines, lifetimes of 128–362 cycles, trajectories of 6 sensors on 3 engines, detection of non-informative sensors).

### Observed
- 7 sensors to drop in FD001: `sensor_1, 5, 6, 10, 16, 18, 19` (constant or 2 values); 14 informative sensors kept.

## v0.0 - initial setup (2026-10-28)

### Added
- Repo structure (`src/` and its sub-packages, `data/processed/`, `notebooks/`, `tests/`, `models/`, `manuals/`).
- Python virtual environment and `requirements.txt` (numpy, pandas, scikit-learn, matplotlib, jupyter, torch).
- `.gitignore`, skeleton `README.md`, `CHANGELOG.md`.
