# Dataset Description

This project uses the English-Bangla portion of the Samanantar parallel corpus from AI4Bharat:

- Hugging Face dataset: `ai4bharat/samanantar`
- Configuration: `bn`
- Source language: English
- Target language: Bangla
- Reported full training split size used by the experiments: 8,604,580 sentence pairs

The final low-resource experiment did not download or train on the full corpus. It streamed a deterministic subset from the dataset and used:

- 5,000 original training sentence pairs
- 500 validation sentence pairs
- 500 test sentence pairs
- 3,000 Bangla sentences as candidate inputs for reverse-model back-translation
- 2,000 filtered synthetic English-Bangla pairs for the final improved model

The measured split statistics are stored in `dataset_statistics.csv`.

Preprocessing steps:

- Removed empty records
- Removed duplicate English-Bangla pairs
- Normalized whitespace
- Used the `google/mt5-small` tokenizer
- Used task prefixes:
  - `translate English to Bengali:` for English-to-Bangla training
  - `translate Bengali to English:` for reverse-model training

Synthetic data filtering removed:

- Empty generated outputs
- Very short generated outputs
- Duplicate synthetic outputs
- Outputs with unsuitable source/target length ratios
- Outputs dominated by punctuation or non-English-like noise
- Repeated generic outputs

