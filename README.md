# English-Bangla Machine Translation with Back-Translation

This repo is for my Master's research project:

**Improving English-Bangla Translation in Low-Resource Settings Using Back-Translation**

The main idea was simple: I wanted to test if back-translation can help English to Bangla translation when we do not use a very large amount of parallel data.

I used:

- `google/mt5-small`
- Samanantar English-Bangla dataset
- Google Colab T4 GPU
- SacreBLEU for evaluation

The final result was not that back-translation improved everything. Actually, in my final stronger experiment, BLEU went down. I kept that result because it is the real measured result.

## Research Question

Does back-translation improve English-Bangla translation performance in a low-resource setting?

My final answer from this experiment:

> In this setup, no. Back-translation reduced BLEU, even after increasing and filtering synthetic data.

It did improve validation loss a little, but BLEU became worse.

## Dataset

I used the English-Bangla part of Samanantar:

- Dataset: `ai4bharat/samanantar`
- Config: `bn`
- Full dataset metadata: `8,604,580` sentence pairs

For the actual low-resource experiment I used:

| Split | Sentence pairs | Avg English length | Avg Bangla length |
|---|---:|---:|---:|
| Train | 5,000 | 11.46 | 10.13 |
| Validation | 500 | 11.81 | 10.40 |
| Test | 500 | 11.83 | 10.73 |
| Raw synthetic BT candidates | 3,000 | 7.23 | 10.29 |
| Filtered synthetic BT pairs used | 2,000 | 7.64 | 10.28 |
| Improved training total | 7,000 | 10.37 | 10.17 |

The dataset statistics file is here:

[`data/dataset_statistics.csv`](data/dataset_statistics.csv)

## What I Did

The experiment had two main models.

### 1. Baseline model

I trained `google/mt5-small` on the original 5,000 English-Bangla sentence pairs.

Input format:

```text
translate English to Bengali: <English sentence>
```

Target:

```text
<Bangla sentence>
```

### 2. Improved model with back-translation

For back-translation, I first trained a reverse model:

```text
Bangla -> English
```

Then I used that reverse model to generate synthetic English from Bangla sentences.

After that, I paired:

```text
synthetic English + original Bangla
```

and trained the English-to-Bangla model again using:

```text
5,000 original pairs + 2,000 filtered synthetic pairs
```

## Synthetic Data Filtering

The first back-translation result was weak, so I ran a stronger final test.

In that final test:

- Raw synthetic candidates: `3,000`
- Filtered synthetic pairs kept: `2,000`

I filtered out:

- empty outputs
- very short outputs
- duplicate outputs
- outputs with bad source/target length ratio
- mostly punctuation/noisy outputs
- repeated generic outputs

Even after this, some synthetic English was still not semantically good. That is probably one main reason BLEU still dropped.

## Final Result

Final stronger filtered back-translation run:

| Model | BLEU | Validation loss | Training loss | Training time |
|---|---:|---:|---:|---:|
| Baseline mT5-small | 0.2359 | 3.1324 | 5.9029 | 735.46 s |
| Reverse mT5-small | N/A | 3.5817 | 6.2214 | 609.03 s |
| Improved filtered BT mT5-small | 0.2114 | 3.0680 | 5.6095 | 940.81 s |

BLEU change:

- Absolute change: `-0.0244`
- Relative change: `-10.36%`

So the honest conclusion is:

> Back-translation did not improve BLEU in this experiment. It slightly improved validation loss, but the generated translations did not match the references better.

## Why I Think BLEU Dropped

The reverse model was also trained with limited data. So when it generated synthetic English, the sentences were sometimes fluent-looking but not accurate enough.

That means the improved model was trained with extra data, but some of that extra data had wrong or weak meaning. More data did not automatically mean better data.

This is the main thing I learned from the project.

## Repo Structure

```text
COMP8851-English-Bangla-BackTranslation/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── dataset_description.md
│   └── dataset_statistics.csv
├── notebooks/
│   ├── baseline_training.ipynb
│   ├── back_translation.ipynb
│   └── evaluation.ipynb
├── src/
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── generate_synthetic_data.py
│   ├── train_improved.py
│   └── evaluate.py
├── results/
│   ├── bleu_scores.csv
│   ├── sample_translations.csv
│   ├── charts/
│   └── experiment_logs/
├── report/
└── docs/
```

## Important Files

- [`notebooks/back_translation.ipynb`](notebooks/back_translation.ipynb)  
  Main final Colab experiment.

- [`src/preprocess.py`](src/preprocess.py)  
  Dataset cleaning and splitting.

- [`src/train_baseline.py`](src/train_baseline.py)  
  Baseline model training.

- [`src/generate_synthetic_data.py`](src/generate_synthetic_data.py)  
  Synthetic data generation and filtering.

- [`src/train_improved.py`](src/train_improved.py)  
  Combines original and synthetic data.

- [`src/evaluate.py`](src/evaluate.py)  
  SacreBLEU evaluation.

- [`results/`](results/)  
  Final measured results, logs, charts, and sample translations.

## How to Run

The easiest way is Colab.

1. Open Google Colab.
2. Upload [`notebooks/back_translation.ipynb`](notebooks/back_translation.ipynb).
3. Set runtime to T4 GPU.
4. Run all cells.
5. Download the output ZIP from Colab.

Local training is possible, but CPU is too slow for this project. GPU is strongly recommended.

Basic local setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare data:

```bash
python src/preprocess.py --total 10500 --out-dir data/processed
```

Train baseline:

```bash
python src/train_baseline.py \
  --train data/processed/train.csv \
  --validation data/processed/validation.csv \
  --output-dir models/baseline
```

The Colab notebook is the main reproducible version for the final results.

## Limitations

Some important limitations:

- I used only 5,000 training sentence pairs to simulate low-resource translation.
- mT5-small is small and not the strongest possible model.
- The reverse model produced noisy synthetic English.
- BLEU is not perfect for judging Bangla translation quality.
- Colab T4 limits training time, memory, and model size.

## Future Work

Things I would try next:

- use more original parallel data
- use a stronger reverse model
- generate more synthetic data but filter it better
- use semantic similarity filtering
- train for more epochs
- compare with mBART or larger mT5 models

## Final Takeaway

Back-translation is useful in many machine translation projects, but in this experiment it did not improve English-Bangla BLEU. The quality of synthetic data mattered more than just increasing the amount of data.

