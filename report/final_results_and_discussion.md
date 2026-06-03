# Final Results and Discussion

## Project Overview

This document reports the final measured results for the Master's project **Improving English-Bangla Translation in Low-Resource Settings Using Back-Translation**. The experiment uses `google/mt5-small` and the Samanantar English-Bangla dataset to test whether adding synthetic back-translated data improves English-to-Bangla translation in a low-resource setting.

All values in this document are taken from generated experiment files. No values are invented. The primary result is the latest valid Colab run located at `/Users/arianzesan/Downloads/mt5_bn_backtranslation_results (2)`.

## Run Audit and Best Run Selection

| Run folder                                                     | Status                         | Evidence                                                                                                                                     | Use in final analysis                    |
|:---------------------------------------------------------------|:-------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------|
| work/colab_results                                             | Invalid older Colab run        | fp16 T4 run had training loss 0.0, validation loss NaN, empty predictions, and 0 synthetic pairs.                                            | Excluded.                                |
| outputs/mt5_backtranslation_scaled_assets                      | Valid but weaker local CPU run | Finite losses and 500 synthetic pairs, but only 24 optimizer steps on CPU; BLEU 0.001961 -> 0.002275.                                        | Background/failed-scale comparison only. |
| /Users/arianzesan/Downloads/mt5_bn_backtranslation_results (2) | Best valid run                 | Tesla T4, fp32, 5,000 train, 500 validation, 500 test, 500 synthetic pairs, finite losses, non-empty predictions, full 3/2/3 epoch schedule. | Primary result.                          |

The latest Colab run is the best valid run because it used a Tesla T4 GPU, disabled fp16 after the earlier numerical failure, completed the intended 3/2/3 epoch schedule, generated 500 usable synthetic pairs, produced finite validation losses, and generated non-empty predictions for both baseline and improved models.

## Table 1: Dataset Statistics

| Metric                                       |   Value |
|:---------------------------------------------|--------:|
| Full Samanantar Bengali train split metadata | 8604580 |
| Training pairs                               |    5000 |
| Validation pairs                             |     500 |
| Test pairs                                   |     500 |
| Bangla monolingual pool                      |     500 |
| Synthetic pairs generated                    |     500 |
| Candidate rows scanned                       |    7500 |
| Empty rows removed                           |       0 |
| Duplicate pairs removed                      |       0 |

Split-level statistics from the latest valid run:

| Split                             |   Sentence pairs | Avg English length (words)               | Avg Bangla length (words)                |
|:----------------------------------|-----------------:|:-----------------------------------------|:-----------------------------------------|
| Full Samanantar bn train metadata |          8604580 | Not computed; full corpus not downloaded | Not computed; full corpus not downloaded |
| Train                             |             5000 | 11.46                                    | 10.13                                    |
| Validation                        |              500 | 11.81                                    | 10.4                                     |
| Test                              |              500 | 11.83                                    | 10.73                                    |
| Bangla monolingual pool           |              500 | 11.68                                    | 10.49                                    |

The full Samanantar Bengali training split contains **8,604,580** sentence pairs according to the dataset metadata. The low-resource simulation used **5,000** training pairs, **500** validation pairs, and **500** test pairs. The back-translation stage requested **500** synthetic pairs and generated **500** usable synthetic pairs.

## Table 2: Training Configuration

| Setting               | Value                                               |
|:----------------------|:----------------------------------------------------|
| Model                 | google/mt5-small                                    |
| Dataset               | ai4bharat/samanantar, bn                            |
| Task                  | English to Bangla translation with back-translation |
| Learning rate         | 0.0001                                              |
| Batch size            | 4                                                   |
| Gradient accumulation | 4                                                   |
| Effective batch size  | 16                                                  |
| Baseline epochs       | 3                                                   |
| Reverse epochs        | 2                                                   |
| Improved epochs       | 3                                                   |
| Optimizer             | Adafactor                                           |
| Precision             | fp32 (FP16=False)                                   |
| Hardware              | Tesla T4                                            |
| Max source length     | 64                                                  |
| Max target length     | 64                                                  |
| BLEU tokenizer        | flores200                                           |

The run used `google/mt5-small` for the forward English-to-Bangla baseline model, the reverse Bangla-to-English model, and the improved English-to-Bangla model. The runtime was **Tesla T4**. Training was run in fp32 (`FP16=False`) because the earlier fp16 Colab run produced NaN losses and empty predictions.

## Table 3: Performance Comparison

| Metric                                      |      Value |
|:--------------------------------------------|-----------:|
| Baseline BLEU                               |   0.302027 |
| Improved BLEU                               |   0.182157 |
| Absolute BLEU change                        |  -0.11987  |
| BLEU change %                               | -39.69     |
| Baseline validation loss                    |   3.1289   |
| Improved validation loss                    |   3.1205   |
| Validation loss change, improved - baseline |  -0.0084   |
| Validation loss change %                    |  -0.27     |
| Baseline training loss                      |   5.7707   |
| Improved training loss                      |   5.8516   |
| Reverse model validation loss               |   3.8535   |

Training-time summary:

| Model              |   Validation loss |   Training loss |   Training time seconds |   Test generation time seconds |
|:-------------------|------------------:|----------------:|------------------------:|-------------------------------:|
| Baseline mT5-small |            3.1289 |          5.7707 |                  703.66 |                          38.56 |
| Reverse mT5-small  |            3.8535 |          7.1992 |                  401.36 |                          30.96 |
| Improved mT5-small |            3.1205 |          5.8516 |                  685.83 |                          39.88 |

### Quantitative Results

The baseline model achieved a BLEU score of **0.3020**. The improved model trained on original plus synthetic back-translated data achieved a BLEU score of **0.1822**. Therefore, BLEU changed by **-0.1199**, or **-39.69%** relative to the baseline.

This means that, in the latest valid run, back-translation **did not improve BLEU**. The improved model's BLEU score was lower than the baseline score.

Validation loss tells a slightly different story. The baseline validation loss was **3.1289**, while the improved model validation loss was **3.1205**. This is a small decrease of **0.0084** (0.27% relative decrease). Lower validation loss is generally better, so the improved model showed a slight validation-loss improvement even though BLEU decreased.

The result is therefore mixed: back-translation slightly improved validation loss but reduced BLEU. Because BLEU is the main translation-quality metric requested for this project, the safest conclusion is that back-translation did **not** clearly improve English-Bangla translation quality in this run.

## Table 4: Translation Examples

| Source English                                                                                                                                                                    | Reference Bangla                                                                                                              | Baseline Prediction                                    | Improved Prediction                            |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------|:-----------------------------------------------|
| He cannot use his right arm.                                                                                                                                                      | তার ডান হাতের কব্জি নেই।                                                                                                       | এবং তাঁর সঙ্গে থাকতে পারবে না।                           | তাঁর সঙ্গে থাকে না।                              |
| Jammu and Kashmir Governor Satyapal Malik dissolved the state assembly, which has been in suspended animation, shortly after rival alliances staked claim to form the government. | সরকার গঠনের দাবি উঠতেই তড়িঘড়ি জম্মু-কাশ্মীর বিধানসভা ভেঙে দিলেন রাজ্যপাল সত্যপাল মালিক।                                            | এবং ভারতের সরকারের সঙ্গে যোগাযোগ করতে পারেনি।           | এবং ভারতের সরকারের সঙ্গে যোগাযোগ করতে পারবে না। |
| The Administration of the University was made aware of this.                                                                                                                      | বিষয়টি বিশ্ববিদ্যালয় প্রশাসনের নজরে এসেছে।                                                                                       | তাঁর সরকারের সঙ্গে যোগাযোগ করেন।                         | তাঁর সঙ্গে যোগাযোগ করা হয়েছে।                    |
| But this notion appears to be false.                                                                                                                                              | তবে এ ধারণা ভুল বলেই মনে হচ্ছে।                                                                                                 | তবে এই যুক্তি ছিল না।                                    | তবে এই যুক্তি ছিল না।                            |
| Public Education:                                                                                                                                                                 | সামাজিকতা শিক্ষা :                                                                                                             | তাঁর শিক্ষার দায়িত্ব:                                     | এবং সরকারের শিক্ষার ব্যবস্থা করা হয়েছে।           |
| It begins in childhood, typically between the ages of 3 and 18 months old.                                                                                                        | এটা শৈশবে বৈশিষ্ট্যগতভাবে শুরু হয় ৩ থেকে ১৮ মাস বয়সের মধ্যে।                                                                       | এবং বয়সের মধ্যে তিন বছর বয়সে জন্ম হয়।                    | তাদের মধ্যে তিন বছর বয়সে জন্ম হয়।                |
| Read what he said.                                                                                                                                                                | পড়ে নিন কি লিখলেন তিনি।                                                                                                       | তাঁর কথা বলেন।                                          | তাঁর সঙ্গে তিনি বলেন।                            |
| So the sell pressure remains always high.                                                                                                                                         | কেননা নতুন ইস্যু এবং শেয়ারসংখ্যা কম হওয়ায় সর্বদাই প্রাথমিক ইস্যুর শেয়ার ক্রয়ের জন্য অতিরিক্ত চাপ পরিলক্ষিত হয়।                            | তবে এর মধ্যে অনেক বড় হচ্ছে।                              | তাঁর মধ্যে এই বাজারের মূল্য বৃদ্ধি পাওয়া যায়।        |
| """Amit Shah and Arun Jailtey are no Gods"                                                                                                                                        | "আশুতোষের কথায়, ""অমিত শাহ আর অরুণ জেটলি ভগবান নয়।"                                                                             | তাঁর মধ্যে তিনি ছিলেন না।                                | তাঁর মধ্যে তিনি ছিলেন না।                        |
| Software architecture refers to the fundamental structures of a software system and the discipline of creating such structures and systems.                                       | সফটওয়্যার আর্কিটেকচার বলতে বোঝায় একটি সফটওয়্যার সিস্টেমের উচ্চ স্তরের কাঠামো, এই ধরনের কাঠামো তৈরি নিয়মাবলি, এবং এই কাঠামোর নথিপত্র। | এবং এর মধ্যে বিভিন্ন নীতি ও প্রযুক্তির নীতির ভিত্তিতে রয়েছে। | এবং এর মধ্যে একটি বড় স্তরের ভিত্তি রয়েছে।         |

## Section A: Quantitative Results

The quantitative evidence is mixed but leans negative for the research question. The validation loss improved slightly from **3.1289** to **3.1205**, suggesting that the improved model may have fit the validation distribution marginally better. However, BLEU decreased from **0.3020** to **0.1822**. Since BLEU directly compares generated Bangla translations with reference Bangla translations, the BLEU result is more relevant to the research question.

The baseline model also had a slightly lower training loss (**5.7707**) than the improved model (**5.8516**). The improved model trained on 5,500 pairs, including 500 synthetic pairs, so its training distribution was different and noisier.

The reverse model validation loss was **3.8535**. The synthetic examples show that the reverse model generated fluent-looking English in some cases, but many synthetic English sentences only loosely match the Bangla target. This synthetic noise likely reduced the usefulness of back-translation.

## Section B: Qualitative Results

The sample translations show partial learning compared with earlier failed runs, but the translations are still weak. Both baseline and improved models often produce short, generic Bangla-like sentences rather than faithful translations.

Some outputs are grammatical fragments. For example, for “But this notion appears to be false,” both models produce `তবে এই যুক্তি ছিল না।`, which is grammatical Bangla-like output but not a precise translation of the reference `তবে এ ধারণা ভুল বলেই মনে হচ্ছে।`.

The improved model sometimes produces slightly cleaner or shorter phrasing, such as `তাঁর সঙ্গে থাকে না।` compared with the baseline's `এবং তাঁর সঙ্গে থাকতে পারবে না।`, but this is not necessarily semantically better because the source was “He cannot use his right arm.” Both predictions miss the key meaning about the right arm.

Common remaining errors include:

- Generic outputs that do not preserve source meaning.
- Missing named entities such as Jammu and Kashmir, Satyapal Malik, Amit Shah, and Arun Jaitley.
- Incorrect semantic substitutions, such as translating institutional or political content into generic government/contact phrases.
- Overly short outputs compared with reference translations.
- Weak handling of technical content, as seen in the software architecture example.

No strong grammar or word-choice improvement can be claimed overall. The improved model had a slightly lower validation loss, but the sample translations and BLEU score do not show improved translation quality.

## Section C: Limitations

### Hardware and Runtime Limitations

The final valid run used a Colab **Tesla T4** GPU, batch size **4**, gradient accumulation **4**, and effective batch size **16**. It used fp32 because fp16 caused an earlier invalid run. This made the run stable but slower and more memory-conscious.

### Dataset Limitations

The experiment used a 5,000-pair low-resource subset from Samanantar rather than the full **8,604,580** pair Bengali training split. This is appropriate for simulating a low-resource setting, but it limits model exposure to English-Bangla translation patterns.

The synthetic dataset contained **500** pairs. Although this meets the minimum target, back-translation often benefits from larger and better-filtered synthetic corpora.

### Training Limitations

The reverse model was trained for **2** epochs. Its synthetic outputs were not manually filtered for semantic adequacy beyond removing empty/sentinel outputs. As a result, some synthetic English sources were noisy or only loosely related to the Bangla targets.

The improved model used original plus synthetic data, but if the synthetic data is noisy, the added examples can hurt sequence-level quality even when validation loss improves slightly. This likely explains why validation loss decreased but BLEU decreased.

### Metric Limitations

BLEU is useful for automatic comparison, but it can be harsh for Bangla and for small test sets. A future report should add chrF and human adequacy/fluency evaluation. Still, because BLEU was the required evaluation metric, the BLEU decrease is important.

## Section D: Future Work

Future work should focus on improving synthetic-data quality and evaluating beyond BLEU:

- Train the reverse Bangla-to-English model for more epochs before generating synthetic data.
- Increase synthetic data from 500 to 1,000 or 2,000 pairs only after verifying reverse-model quality.
- Add filtering for synthetic pairs using length ratio, language identification, duplicate removal, and semantic similarity checks.
- Train on 10,000 and 20,000 original pairs with the stable fp32/Adafactor setup if Colab runtime permits.
- Try larger or translation-specialized multilingual models, including mBART, NLLB, IndicBART, or larger mT5 variants.
- Add chrF, COMET-style metrics if available, and human evaluation for adequacy and fluency.
- Save model checkpoints so that qualitative error analysis can be performed on more examples.

## Final Conclusion

The latest valid Colab experiment does **not** provide clear evidence that back-translation improved English-Bangla translation performance. The improved model had a slightly better validation loss (**3.1205** vs **3.1289**), but BLEU decreased from **0.3020** to **0.1822**. The sample translations also show that both models often generated generic or semantically incorrect Bangla outputs.

The best-supported conclusion is: **with the current mT5-small setup, 5,000 original training pairs, and 500 synthetic back-translated pairs, back-translation did not improve BLEU-based English-Bangla translation performance. Better reverse-model training, more synthetic data, and stronger filtering are needed before back-translation can be expected to help.**

## Generated Final Assets

- `figures/bleu_comparison_chart.png`
- `figures/validation_loss_comparison_chart.png`
- `figures/training_loss_chart.png`
- `figures/dataset_distribution_chart.png`
- `tables/table_1_dataset_statistics.csv`
- `tables/table_2_training_configuration.csv`
- `tables/table_3_performance_comparison.csv`
- `tables/table_4_translation_examples.csv`
- `tables/training_time.csv`
- `tables/synthetic_examples_sample.csv`
- `tables/run_audit.csv`
