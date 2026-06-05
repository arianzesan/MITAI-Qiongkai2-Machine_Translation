# Final Results and Discussion

## Project Overview

This document reports the final measured results for the Master's project **Improving English-Bangla Translation in Low-Resource Settings Using Back-Translation**.

The experiment tested whether back-translation improves English-to-Bangla translation in a low-resource setting using:

- `google/mt5-small`
- Samanantar English-Bangla dataset
- Google Colab Tesla T4 GPU
- SacreBLEU with the `flores200` tokenizer

No result values in this document are invented. The final values come from the stronger filtered back-translation experiment in `final_stronger_bt_experiment`.

## Run Audit

| Run | Status | Reason |
|---|---|---|
| Earlier CPU/device run | Valid but too weak | CPU-bound and only a small number of optimizer steps. Used as background only. |
| Earlier Colab run with 500 synthetic pairs | Valid but weaker | Used T4 GPU and finite losses, but synthetic data was small. BLEU dropped from 0.3020 to 0.1822. |
| Final stronger filtered BT run | Final reported run | Used the same 5,000/500/500 split, generated 3,000 synthetic candidates, filtered them, and trained with 2,000 synthetic pairs. |

The final stronger run is the main reported run because it directly tested whether the earlier BLEU drop was caused by too little or too noisy synthetic data.

## Table 1: Dataset Statistics

| Split | Sentence pairs | Avg English length (words) | Avg Bangla length (words) |
|---|---:|---:|---:|
| Full Samanantar bn train metadata | 8,604,580 | Not computed; full corpus not downloaded | Not computed; full corpus not downloaded |
| Train | 5,000 | 11.46 | 10.13 |
| Validation | 500 | 11.81 | 10.40 |
| Test | 500 | 11.83 | 10.73 |
| BT candidate monolingual pool | 3,000 | 11.68 | 10.29 |
| Raw synthetic BT candidates | 3,000 | 7.23 | 10.29 |
| Filtered synthetic BT pairs used | 2,000 | 7.64 | 10.28 |
| Improved training total | 7,000 | 10.37 | 10.17 |

The full Samanantar Bengali train split has **8,604,580** sentence pairs according to dataset metadata. The full dataset was not uploaded to GitHub because it is too large. The experiment streamed and used a low-resource subset.

## Table 2: Training Configuration

| Setting | Value |
|---|---|
| Model | `google/mt5-small` |
| Dataset | `ai4bharat/samanantar`, Bengali/Bangla shard |
| Hardware | Google Colab Tesla T4 |
| Optimizer | Adafactor |
| Learning rate | 0.0001 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Baseline epochs | 3 |
| Reverse model epochs | 3 |
| Improved model epochs | 3 |
| Precision | fp32 |
| Max source length | 128 |
| Max target length | 128 |
| BLEU tokenizer | `flores200` |

fp32 was used because an earlier fp16 run produced numerical instability, including NaN validation loss and empty predictions.

## Table 3: Performance Comparison

| Model | BLEU | Validation loss | Training loss | Training time |
|---|---:|---:|---:|---:|
| Baseline mT5-small | 0.2359 | 3.1324 | 5.9029 | 735.46 s |
| Reverse mT5-small | N/A | 3.5817 | 6.2214 | 609.03 s |
| Improved filtered BT mT5-small | 0.2114 | 3.0680 | 5.6095 | 940.81 s |

BLEU changed from **0.2359** to **0.2114**.

- Absolute BLEU change: **-0.0244**
- Relative BLEU change: **-10.36%**
- Validation loss change: **3.1324 -> 3.0680**

The improved model had lower validation loss, but lower BLEU.

## Table 4: Translation Examples

| Source English | Reference Bangla | Baseline Prediction | Improved Prediction | Better / Worse note |
|---|---|---|---|---|
| He cannot use his right arm. | তার ডান হাতের কব্জি নেই। | এবং তিনি নিজেদের সঙ্গে থাকতে পারবে না। | তাঁর সঙ্গে থাকা যায়নি। | Worse |
| Jammu and Kashmir Governor Satyapal Malik dissolved the state assembly, which has been in suspended animation, shortly after rival alliances staked claim to form the government. | সরকার গঠনের দাবি উঠতেই তড়িঘড়ি জম্মু-কাশ্মীর বিধানসভা ভেঙে দিলেন রাজ্যপাল সত্যপাল মালিক। | তাঁর বিরুদ্ধে সরকারের সঙ্গে যোগাযোগ করেন। | তাঁর বিরুদ্ধে সরকারের সঙ্গে যোগাযোগ করতে পারেনি। | Better, but still misses key named entities |
| The Administration of the University was made aware of this. | বিষয়টি বিশ্ববিদ্যালয় প্রশাসনের নজরে এসেছে। | এবং সরকারের সঙ্গে যোগাযোগ করা হয়েছে। | তাঁর সঙ্গে যোগাযোগ করা হয়েছে। | Worse |
| But this notion appears to be false. | তবে এ ধারণা ভুল বলেই মনে হচ্ছে। | তবে এটা কিন্তু এই নীতিমালা করা হয়েছে। | তবে এই বিষয়টি প্রকাশ করতে পারেনি। | Better, but still not faithful |
| Public Education: | সামাজিকতা শিক্ষা : | এবং সরকারের নীতিমালা: | তাঁর দায়িত্ব: | No clear change |
| It begins in childhood, typically between the ages of 3 and 18 months old. | এটা শৈশবে বৈশিষ্ট্যগতভাবে শুরু হয় ৩ থেকে ১৮ মাস বয়সের মধ্যে। | তাদের মধ্যে তিন বছর বয়সে জন্ম হয়। | তাঁর জন্মের মধ্যে তিন বছর বয়সে জন্ম হয়। | Slightly better by sentence BLEU, but semantically weak |
| Read what he said. | পড়ে নিন কি লিখলেন তিনি। | তাঁর কথা বলেন। | তাঁর সঙ্গে তিনি বলেন। | Better by sentence BLEU, but still weak |
| So the sell pressure remains always high. | কেননা নতুন ইস্যু এবং শেয়ারসংখ্যা কম হওয়ায় সর্বদাই প্রাথমিক ইস্যুর শেয়ার ক্রয়ের জন্য অতিরিক্ত চাপ পরিলক্ষিত হয়। | তবে এর মধ্যে তারা বেশ বড়। | তাঁর মধ্যে বেশ বড় হচ্ছে। | Worse |
| "Amit Shah and Arun Jailtey are no Gods" | "আশুতোষের কথায়, ""অমিত শাহ আর অরুণ জেটলি ভগবান নয়।" | এবং তিনি ছিলেন না। | তাঁর মধ্যে তিনি ছিলেন না। | No clear change |
| Software architecture refers to the fundamental structures of a software system and the discipline of creating such structures and systems. | সফটওয়্যার আর্কিটেকচার বলতে বোঝায় একটি সফটওয়্যার সিস্টেমের উচ্চ স্তরের কাঠামো, এই ধরনের কাঠামো তৈরি নিয়মাবলি, এবং এই কাঠামোর নথিপত্র। | এবং এর মধ্যে একটি প্রধান স্তম্ভ করা হয়। | তবে, এর মধ্যে প্রযুক্তির নীতিমালা রয়েছে। | Better by sentence BLEU, but still incomplete |

## Section A: Quantitative Results

The final stronger experiment did **not** show a BLEU improvement from back-translation. The baseline model scored **0.2359 BLEU**, while the improved model trained with filtered synthetic data scored **0.2114 BLEU**.

This means back-translation reduced BLEU by **0.0244**, or **10.36%** relative to the baseline.

Validation loss moved in the opposite direction. The baseline validation loss was **3.1324**, while the improved model validation loss was **3.0680**. This suggests the improved model fit the validation distribution slightly better, but the generated translations were not closer to the reference translations according to BLEU.

The main conclusion is that adding filtered synthetic data helped the training objective slightly but did not improve sequence-level translation quality.

## Section B: Qualitative Results

The translation examples show that both models often produced Bangla-like output, but the meaning was frequently incomplete or generic.

Common observations:

- Some improved predictions were shorter and cleaner.
- Some improved predictions had slightly better sentence-level BLEU.
- Both models often missed named entities, such as Jammu and Kashmir, Satyapal Malik, Amit Shah, and Arun Jaitley.
- Both models struggled with long political and technical sentences.
- The improved model still produced generic phrases like government/contact-related outputs that did not match the source meaning.

So it is not safe to claim that grammar or word choice clearly improved overall. The improved model had a lower validation loss, but the sample translations and BLEU score show that translation quality did not improve enough.

## Section C: Limitations

The experiment was limited by Colab hardware and runtime. `google/mt5-small` is trainable on a Tesla T4, but training larger datasets, larger models, and stronger reverse models would need more time and better GPU access.

The project used a low-resource subset of Samanantar rather than the full 8.6 million-pair dataset. This was intentional for the research question, but it limited how much translation knowledge the model could learn.

The biggest limitation was synthetic data quality. The reverse Bangla-to-English model was also trained in a low-resource setup. Because of that, it sometimes generated synthetic English that was fluent-looking but not faithful to the Bangla sentence. Filtering removed obvious bad outputs, but it could not fully remove semantic noise.

BLEU is also limited. It is useful for automatic comparison, but it does not capture every acceptable Bangla translation. Future work should include chrF and human evaluation.

## Section D: Future Work

Future work should focus on improving synthetic data before adding more of it:

- Train the reverse Bangla-to-English model on more data.
- Generate a larger synthetic pool and keep only the best synthetic pairs.
- Use semantic filtering or quality estimation before improved-model training.
- Try 10,000 to 20,000 original training pairs if GPU time allows.
- Train for more epochs with checkpoint selection based on validation BLEU.
- Compare `google/mt5-small` with mBART, NLLB, IndicBART, or larger mT5 variants.
- Add chrF and human adequacy/fluency evaluation.

## Final Conclusion

The final stronger filtered back-translation experiment does **not** prove that back-translation improved English-Bangla translation in this low-resource setup.

The final measured result is:

- Baseline BLEU: **0.2359**
- Improved BLEU: **0.2114**
- Baseline validation loss: **3.1324**
- Improved validation loss: **3.0680**

Back-translation reduced BLEU but improved validation loss. The most likely reason is that the synthetic English sources were still noisy, even after filtering. Therefore, the honest final conclusion is that back-translation was tested properly, but it did not improve BLEU for this specific mT5-small, low-resource English-Bangla setup.

## Generated Assets

The final repository includes:

- `results/bleu_scores.csv`
- `results/sample_translations.csv`
- `results/charts/bleu_comparison_chart.png`
- `results/charts/validation_loss_chart.png`
- `results/charts/training_loss_chart.png`
- `results/experiment_logs/training_configuration.csv`
- `results/experiment_logs/validation_loss_comparison.csv`
- `results/experiment_logs/final_stronger_manifest.json`
