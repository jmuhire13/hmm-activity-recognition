# Human activity recognition with a Hidden Markov Model

This project infers four activities (standing, walking, jumping and still) from the
accelerometer and gyroscope of a phone. The data was recorded with the Sensor Logger app on an
iPhone 8 at a nominal 50 Hz (measured interval 20.03 ms). The Hidden Markov Model is written
from scratch with numpy.

The project is in progress. The data preparation, feature extraction, normalization, training of
the HMM and the evaluation on two unseen recordings are finished. The report is still to come.

## Folder layout

| Folder | Content |
| --- | --- |
| `data/raw/` | The original Sensor Logger exports, one folder per activity, never edited. The `Metadata.csv` files (they contain the phone's device identifier) are not published |
| `data/processed/` | 92 labeled windows of 255 readings (5.1 s), 23 per activity, one CSV file each |
| `data/features/` | Feature table, normalized feature table and the saved scaler (`scaler.json`) |
| `data/raw_unseen/` | Two new test recordings, each moving through all four activities (`test_take_1_still_first`, `test_take_2_jumping_first`), the answer key with the time of each activity (`answer_key.csv`), and two short extra takes that are not scored (`demo_take_1`, `demo_take_2`) |
| `src/` | `gaussian_hmm.py`, the hidden Markov model (forward, backward, Viterbi, Baum-Welch) in numpy, and `window_features.py`, the feature and scaling functions applied to new recordings |
| `tests/` | Checks of the model against brute-force calculations, and of the feature module against the saved training features |
| `results/` | Metrics and the prediction for every test window, written by the evaluation notebook |
| `models/` | The trained model parameters (`hmm_parameters.json`) |
| `notebooks/` | The notebooks, run in the order below |
| `figures/` | Figures produced by the notebooks |

## How to reproduce

Install the packages, then run the notebooks in this order from inside the `notebooks/` folder.
The order matters because each notebook reads the files written by the one before it.

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace data_preparation.ipynb
jupyter nbconvert --to notebook --execute --inplace feature_extraction.ipynb
jupyter nbconvert --to notebook --execute --inplace normalization.ipynb
jupyter nbconvert --to notebook --execute --inplace model_design_checks.ipynb
jupyter nbconvert --to notebook --execute --inplace hmm_training.ipynb
jupyter nbconvert --to notebook --execute --inplace evaluation.ipynb
```

To run the checks, go back to the repository folder and run
`python tests/test_gaussian_hmm.py` and `python tests/test_window_features.py`.

1. `data_preparation.ipynb` merges the accelerometer and gyroscope files, trims the first 5 s of
   each recording, cuts the rest into windows and saves them to `data/processed/`.
2. `feature_extraction.ipynb` computes six features per sensor for every window and writes
   `data/features/window_features.csv`.
3. `normalization.ipynb` applies a logarithm to the strongly skewed features, standardizes all
   features with a Z-score and saves the scaler used, so new recordings can be scaled the same way.
4. `model_design_checks.ipynb` compares four feature sets by blocked cross-validation inside the
   training windows and measures how much the assumed probability of staying in the same activity
   matters. It leads to the use of eight of the twelve features.
5. `hmm_training.ipynb` trains the HMM on those eight features with Baum-Welch, plots the
   convergence of the log likelihood, and saves the trained parameters to
   `models/hmm_parameters.json`.
6. `evaluation.ipynb` cuts the two test recordings into windows, computes and scales the features
   with the saved training scaler, decodes each recording with Viterbi, grades the windows against
   `data/raw_unseen/answer_key.csv` and reports sensitivity, specificity, accuracy and the
   confusion matrix. It also draws the transition matrix, the emission distributions and the
   decoded sequences.
