# Model Comparison Metrics

## Average Confidence Scores (IR Score)
| Model | Avg Confidence |
|---|---|
| distilbert | 0.4183 |
| roberta | 0.1683 |
| aerobert | 0.0046 |

## Inter-Model Agreement (Relative to RoBERTa)
Since we lack ground truth, we measure how much other models agree with RoBERTa (SQuAD2 strong baseline).

| Comparison | Avg F1 Agreement |
|---|---|
| agreement_roberta_distilbert_f1 | 0.4520 |
| agreement_roberta_aerobert_f1 | 0.1716 |
