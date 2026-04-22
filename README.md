# Multiple Disease Prediction System

This project is a Streamlit application that predicts:

- Diabetes
- Heart disease
- Parkinson's disease
- Kidney disease

The repository now has a single local workflow for both inference and retraining:

- `MultipleDiseasePridiction_app.py` is the app entry point.
- `train_models.py` is the canonical local training script.
- `project_config.py` stores shared paths, model filenames, and UI input definitions.
- `Src/*.ipynb` remain useful as exploratory notebooks, but they are no longer the source of truth for local execution.

## Project Structure

```text
Project/
├── Datasets/
├── Src/
│   ├── *.ipynb
│   └── *.sav
├── MultipleDiseasePridiction_app.py
├── project_config.py
├── train_models.py
├── requirements.txt
└── home.gif
```

## Environment

This project already includes a local virtual environment named `mdp`.

Use it from inside the `Project` folder:

```bash
cd /home/thearchfiend/MultipleDiseasePrediction-main/MultipleDiseasePrediction-main/Project
./mdp/bin/python --version
```

If you need to reinstall dependencies:

```bash
./mdp/bin/pip install -r requirements.txt
```

## Run The App

Start Streamlit with:

```bash
./mdp/bin/streamlit run MultipleDiseasePridiction_app.py
```

By default, Streamlit will print a local URL such as:

```text
http://127.0.0.1:8501
```

## Retrain The Models

The app depends on four saved model files inside `Src/`.

Regenerate them with:

```bash
./mdp/bin/python train_models.py
```

This creates or refreshes:

- `Src/diabetes_prediction_model.sav`
- `Src/heart_prediction_model.sav`
- `Src/parkinsons_prediction_model.sav`
- `Src/kidney_prediction_model.sav`

## What Changed

The project was cleaned up in three areas:

- UI and validation:
  - free-text prediction inputs were replaced with validated `number_input` and `selectbox` controls
  - binary encodings are now handled by the UI instead of requiring users to remember `0` and `1`
  - forms prevent partial reruns while typing
- Documentation:
  - this README now includes setup, run, and retraining instructions
  - the repo structure and canonical workflow are documented
- Consistency:
  - model paths and filenames are defined once in `project_config.py`
  - the kidney app inputs now match the six features used by the local training script
  - app and training logic no longer depend on the old Colab-only notebook paths

## Notebook Notes

The notebooks in `Src/` were originally authored for Google Colab and reference Drive paths. They are still helpful for experimentation, but for this local project:

- use `train_models.py` to build models
- use `MultipleDiseasePridiction_app.py` to run the interface
- use `project_config.py` when changing model filenames or input definitions

## Known Limitations

- These models are trained on the datasets bundled with the project and should be treated as educational/demo predictions, not medical advice.
- The heart and Parkinson's screens still rely on encoded clinical categories from the source datasets.
- The training code intentionally follows the existing project approach rather than redesigning the modeling pipeline from scratch.
