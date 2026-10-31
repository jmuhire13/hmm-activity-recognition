# Human activity recognition with a Hidden Markov Model

This project infers four activities (standing, walking, jumping and still) from the accelerometer
and gyroscope of a phone. The data was recorded by the author with the Sensor Logger app on an
iPhone 8 at a nominal 50 Hz (measured interval 20.03 ms, 49.91 to 49.92 Hz) on 27 October 2026.
The Hidden Markov Model (forward, backward, Viterbi and Baum-Welch with a log-likelihood
convergence check) is written from scratch with numpy.

## Main result and its limits

On two new recordings that were not used for training, all 47 graded windows were classified
correctly (still 10, standing 5, walking 15, jumping 17): sensitivity, specificity and accuracy are
1.00 for every activity. The result needs careful reading:

- only 9 of the 47 windows are independent (the other windows come from shifted cuts of the same
  two recordings), and standing has a single independent window;
- the test is close to the training conditions (same person, phone and app, three days later);
- classifying every window alone gives the same decisions as the full model on all graded windows,
  so the test does not show that the transition matrix helps;
- the training recordings contain no switch between activities, so the transition matrix is mostly
  an assumption (about 90 percent probability of staying in an activity, before the data is added).

The report discusses these limits in detail.

## Folder layout

| Folder | Content |
| --- | --- |
| `data/raw/` | The original Sensor Logger exports, one folder per activity, never edited. The `Metadata.csv` files (they contain the phone's device identifier) are not published |
| `data/processed/` | 92 labeled windows of 255 readings (5.1 s), 23 per activity, one CSV file each, with the columns `time_s`, `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z` and `activity` |
| `data/features/` | Feature table, normalized feature table and the saved scaler (`scaler.json`) |
| `data/raw_unseen/` | Two test recordings recorded on 30 October 2026, each moving through all four activities (`test_take_1_still_first`, `test_take_2_jumping_first`), the answer key with the time of each activity (`answer_key.csv`), and two short extra takes that are not scored (`demo_take_1`, `demo_take_2`) |
| `src/` | `gaussian_hmm.py`, the hidden Markov model in numpy, and `window_features.py`, the feature and scaling functions applied to new recordings |
| `tests/` | Checks of the model against brute-force calculations, and of the feature module against the saved training features |
| `results/` | Metrics and the prediction for every test window, written by the evaluation notebook |
| `models/` | The trained model parameters (`hmm_parameters.json`) |
| `notebooks/` | The notebooks, run in the order below |
| `figures/` | Figures produced by the notebooks |

Units, as stated in the Sensor Logger documentation: the accelerometer is in m/s² (gravity removed) and
the gyroscope in rad/s.

## How to reproduce

The code was run with Python 3.13.3. Install the packages, then run the notebooks in this order
from inside the `notebooks/` folder. The order matters because each notebook reads the files
written by the one before it, and the notebooks write every file in `data/processed/`,
`data/features/`, `models/`, `results/` and `figures/`.

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

This order was checked on 30 October 2026 in a fresh virtual environment installed only from
`requirements.txt`: all six notebooks ran, both test files passed (18 and 3 tests), and every file in
`data/processed/`, `data/features/`, `models/` and `results/` was written again byte for byte
identical to the files in this repository.

If the `jupyter` command is not found, use `python -m jupyter nbconvert ...` instead. To run the
checks, go back to the repository folder and run `python tests/test_gaussian_hmm.py` and
`python tests/test_window_features.py`.

1. `data_preparation.ipynb` merges the accelerometer and gyroscope files, shows the sampling
   interval and the start of each recording, trims the first 5 s, cuts the rest into windows,
   checks them and saves them to `data/processed/`.
2. `feature_extraction.ipynb` computes six features per sensor for every window, compares them
   (correlation, separation between activities), tests which features change when the phone is
   rotated, and writes `data/features/window_features.csv`.
3. `normalization.ipynb` applies a logarithm to the strongly skewed features, standardizes all
   features with a Z-score, saves the scaler used (so new recordings are scaled the same way), and
   tests what the normalization changes in the model.
4. `model_design_checks.ipynb` compares four feature sets by blocked cross-validation inside the
   training windows, measures how much the assumed probability of staying in the same activity
   matters, shows which activities are confused, and tests the effect of a lower sampling rate. It
   leads to the use of eight of the twelve features.
5. `hmm_training.ipynb` trains the HMM on those eight features with Baum-Welch, plots the
   convergence of the log likelihood, and saves the trained parameters to
   `models/hmm_parameters.json`.
6. `evaluation.ipynb` cuts the two test recordings into windows, computes and scales the features
   with the saved training scaler, decodes each recording with Viterbi, grades the windows against
   `data/raw_unseen/answer_key.csv` and reports sensitivity, specificity, accuracy and the
   confusion matrix. It also draws the transition matrix, the emission distributions and the
   decoded sequences, and measures how typical the test windows are of their activity.

Every notebook explains its output in the text after each result.

## Videos

The four videos of the data collection (one per activity) are shared with school accounts only.

- [Standing](https://drive.google.com/file/d/1nf3LvXP8BCD5rzq5eoegU79u1JOaruzC/view?usp=sharing)
- [Walking](https://drive.google.com/file/d/1LBg0yw7PxkivbE484DHloDt-tYCJLcWY/view?usp=sharing)
- [Jumping](https://drive.google.com/file/d/1t4Vv_uH4r1BxB5WdF2alNabtPU8qHBLZ/view?usp=sharing)
- [Still](https://drive.google.com/file/d/1ccBnDAhzgUSlXoFUunQUlOcqW2AATry0/view?usp=sharing)
