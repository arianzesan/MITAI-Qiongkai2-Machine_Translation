# Methodology

This file explains the method I followed for the project.

## Research Question

Does back-translation improve English-Bangla translation performance in a low-resource setting?

## Dataset

I used the English-Bangla part of the Samanantar dataset:

- Dataset name: `ai4bharat/samanantar`
- Config: `bn`
- Full dataset metadata: 8,604,580 sentence pairs

Because the project is about low-resource translation, I did not train on the full dataset. I used a smaller subset:

- 5,000 training pairs
- 500 validation pairs
- 500 test pairs

The split was deterministic, using seed 42.

## Model

I used `google/mt5-small` for all models.

I chose this because:

- it supports multiple languages
- it can work with Bangla
- it is small enough to train on Google Colab T4

## Baseline Training

The baseline model was trained directly on original English-Bangla parallel data.

Input format:

```text
translate English to Bengali: <English sentence>
```

Output target:

```text
<Bangla sentence>
```

This gave the baseline result for comparison.

## Back-Translation

For back-translation, I trained a reverse model:

```text
Bangla -> English
```

Then I used this reverse model to translate Bangla sentences into synthetic English.

So one synthetic training example looked like:

```text
synthetic English sentence -> original Bangla sentence
```

The purpose was to add extra training data for the English-to-Bangla model.

## Stronger Final Experiment

The earlier back-translation result was weak, so I ran one stronger experiment.

In the stronger run:

- raw synthetic candidates: 3,000
- filtered synthetic pairs kept: 2,000
- final improved training set: 5,000 original + 2,000 synthetic = 7,000 pairs

The filtering removed:

- empty outputs
- very short outputs
- duplicate outputs
- outputs with bad length ratio
- punctuation-heavy outputs
- repeated generic outputs
- non-English-like noisy outputs

## Training Setup

Final setup:

- Model: `google/mt5-small`
- GPU: Tesla T4
- Optimizer: Adafactor
- Learning rate: 0.0001
- Batch size: 4
- Gradient accumulation: 4
- Effective batch size: 16
- Epochs: 3 baseline, 3 reverse, 3 improved
- Max source length: 128
- Max target length: 128
- Precision: fp32

I used fp32 because an earlier fp16 run gave NaN loss and empty predictions.

## Evaluation

I evaluated the baseline and improved model on the same 500 test sentences.

Metric:

- SacreBLEU
- Tokenizer: `flores200`

I also saved:

- training loss
- validation loss
- training time
- sample translations
- charts

## Final Result

Final measured result:

| Model | BLEU | Validation loss |
|---|---:|---:|
| Baseline | 0.2359 | 3.1324 |
| Improved with filtered BT | 0.2114 | 3.0680 |

So the improved model had lower validation loss, but worse BLEU.

My conclusion:

> Back-translation did not improve English-Bangla BLEU in this setup.

The likely reason is synthetic data quality. The generated English sentences were sometimes not faithful to the Bangla meaning. Filtering helped remove obvious bad outputs, but it could not fully fix semantic noise.

