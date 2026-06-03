# Final Discussion Notes

This stronger test used the same 5,000/500/500 train/validation/test split as the previous valid Colab run. Baseline was retrained in this stronger notebook because no previous valid baseline folder was found. It increased synthetic generation to 3000 candidates and retained 2000 filtered synthetic pairs for improved-model training.

Baseline BLEU was 0.2359. Improved BLEU was 0.2114. The BLEU change was -0.0244, or -10.36%.

Baseline validation loss was 3.1324. Improved validation loss was 3.0680. The validation-loss change was -0.0644.

Conclusion: the stronger filtered back-translation setup did not improve BLEU. If BLEU still drops, likely causes include noisy synthetic sources from the reverse model, domain mismatch in synthetic pairs, and insufficient reverse-model quality even after filtering.
