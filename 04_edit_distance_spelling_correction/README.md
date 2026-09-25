# 4. Edit Distance and Spelling Correction from Scratch

This topic begins with a dynamic-programming edit-distance table and then builds a small noisy-channel spelling corrector. A misspelled word produces candidate corrections through insert, delete, substitute, and transpose operations. A unigram language model scores how plausible each candidate is.

## Dataset

The download script targets Kaggle's [Misspelled Words dataset](https://www.kaggle.com/datasets/fazilbtopal/misspelled-words). It contains misspelled words paired with their correct labels. The loader detects common column names such as `misspelled`, `wrong`, `correct`, and `label`, then falls back to the first two columns.

The implementation also runs on a small built-in set of pairs. The Kaggle file is kept outside Git through `.gitignore`.

## Mathematics

The Levenshtein dynamic-programming recurrence is:

`D[i,j] = min(D[i-1,j] + 1, D[i,j-1] + 1, D[i-1,j-1] + [source_i != target_j])`

The noisy-channel score used here is:

`score(candidate | observed) = log P(candidate) - penalty * edit_distance(observed, candidate)`

The unigram language model uses add-alpha smoothing:

`P(word) = (count(word) + alpha) / (total_words + alpha * vocabulary_size)`

The best candidate is the one with the greatest score among known vocabulary words within the configured edit distance.

## Code map

- `edit_distance.py` builds the DP table and generates one-edit candidates.
- `language_model.py` implements an add-alpha unigram language model.
- `spelling_corrector.py` combines candidate generation, distance, and language-model scoring.
- `from_scratch.py` loads correction pairs, evaluates exact correction accuracy, and prints examples.
- `baseline_library.py` uses `pyspellchecker` for a library comparison.
- `compare_results.py` compares exact accuracy and runtime.
- `data/download_data.py` downloads the Kaggle dataset.

The language model is intentionally unigram-based until Topic 5 introduces the full n-gram model. Topic 5 can later replace this scorer without changing candidate generation.

## Run in Codespaces

```bash
cd /workspaces/NLP_REC/04_edit_distance_spelling_correction
python -m pip install -r requirements.txt
python from_scratch.py
python compare_results.py
```

Download the Kaggle data:

```bash
python data/download_data.py
python from_scratch.py --csv "data/misspelled_words.csv" --limit 1000
python compare_results.py --csv "data/misspelled_words.csv" --limit 1000
```

If the downloaded filename differs, run `ls data` and pass the actual CSV path to `--csv`.

## What to inspect

- The DP table shows exactly how each edit contributes to the final distance.
- Candidate generation is finite because the corrector only considers known words.
- A distance-only system can choose a frequent but semantically wrong word.
- Language-model probability helps prefer common words when several candidates have the same distance.
- The library baseline is optimized and uses a larger built-in dictionary, so it can outperform the small dataset-trained vocabulary.
