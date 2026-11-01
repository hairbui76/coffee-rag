# System table (overall)

| System | Intent | Runs | Context Precision | Context Recall (Ragas) | Faithfulness | Answer Relevancy |
|---|---|---|---|---|---|---|
| MiniLM | all | 1 | 0.540 | 0.296 | 0.785 | 0.607 |
| MiniLM+RRF | all | 1 | 0.663 | 0.429 | 0.799 | 0.676 |
| BGE-M3 | all | 1 | 0.690 | 0.447 | 0.790 | 0.648 |
| BGE-M3+RRF | all | 1 | 0.819 | 0.582 | 0.800 | 0.693 |

# System table (by intent)

| System | Intent | Runs | Context Precision | Context Recall (Ragas) | Faithfulness | Answer Relevancy |
|---|---|---|---|---|---|---|
| MiniLM | product_search | 1 | 0.189 | 0.105 | 0.787 | 0.580 |
| MiniLM | similar_search | 1 | 0.743 | 0.243 | 0.699 | 0.663 |
| MiniLM | news_search | 1 | 0.689 | 0.542 | 0.870 | 0.578 |
| MiniLM+RRF | product_search | 1 | 0.434 | 0.418 | 0.815 | 0.718 |
| MiniLM+RRF | similar_search | 1 | 0.738 | 0.261 | 0.688 | 0.616 |
| MiniLM+RRF | news_search | 1 | 0.816 | 0.607 | 0.892 | 0.695 |
| BGE-M3 | product_search | 1 | 0.268 | 0.177 | 0.768 | 0.560 |
| BGE-M3 | similar_search | 1 | 0.930 | 0.542 | 0.705 | 0.754 |
| BGE-M3 | news_search | 1 | 0.871 | 0.622 | 0.896 | 0.630 |
| BGE-M3+RRF | product_search | 1 | 0.641 | 0.550 | 0.802 | 0.643 |
| BGE-M3+RRF | similar_search | 1 | 0.925 | 0.547 | 0.690 | 0.773 |
| BGE-M3+RRF | news_search | 1 | 0.892 | 0.650 | 0.908 | 0.664 |

# Paired tests vs BGE-M3 (per-case means across runs, bootstrap 95% CI, Wilcoxon signed-rank)

| System | Metric | n | Δ mean | 95% CI | Wilcoxon p | W/L/T |
|---|---|---|---|---|---|---|
| MiniLM vs BGE-M3 | Context Precision | 600 | -0.149 | [-0.177, -0.120] | <0.001 | 96/295/209 |
| MiniLM vs BGE-M3 | Context Recall (Ragas) | 600 | -0.151 | [-0.180, -0.120] | <0.001 | 79/219/302 |
| MiniLM vs BGE-M3 | Faithfulness | 598 | -0.005 | [-0.026, +0.016] | 0.692 | 206/234/158 |
| MiniLM vs BGE-M3 | Answer Relevancy | 600 | -0.041 | [-0.074, -0.006] | 0.718 | 305/250/45 |
| MiniLM+RRF vs BGE-M3 | Context Precision | 600 | -0.027 | [-0.058, +0.006] | 0.040 | 201/249/150 |
| MiniLM+RRF vs BGE-M3 | Context Recall (Ragas) | 600 | -0.018 | [-0.053, +0.017] | 0.252 | 142/167/291 |
| MiniLM+RRF vs BGE-M3 | Faithfulness | 597 | +0.009 | [-0.011, +0.030] | 0.398 | 220/217/160 |
| MiniLM+RRF vs BGE-M3 | Answer Relevancy | 600 | +0.028 | [-0.003, +0.061] | <0.001 | 327/247/26 |
| BGE-M3+RRF vs BGE-M3 | Context Precision | 600 | +0.130 | [+0.103, +0.156] | <0.001 | 234/88/278 |
| BGE-M3+RRF vs BGE-M3 | Context Recall (Ragas) | 600 | +0.135 | [+0.106, +0.164] | <0.001 | 189/63/348 |
| BGE-M3+RRF vs BGE-M3 | Faithfulness | 597 | +0.011 | [-0.008, +0.031] | 0.321 | 228/203/166 |
| BGE-M3+RRF vs BGE-M3 | Answer Relevancy | 600 | +0.045 | [+0.023, +0.067] | 0.159 | 255/245/100 |

# Metric table (one system, several recall definitions)

System: BGE-M3+RRF

| Intent | Context Recall (Ragas) | Soft Recall@k | Context Precision |
|---|---|---|---|
| product_search | 0.550 | 0.617 | 0.632 ± 0.013 |
| similar_search | 0.547 | 0.712 | 0.929 ± 0.005 |
| news_search | 0.650 | 0.828 | 0.890 ± 0.002 |
| all | 0.582 | 0.719 | 0.817 ± 0.003 |

Spearman ρ between metrics across cases:

| A | B | n | ρ | p |
|---|---|---|---|---|
| Context Recall (Ragas) | Soft Recall@k | 600 | 0.907 | <0.001 |
| Context Recall (Ragas) | Context Precision | 600 | 0.283 | <0.001 |
| Soft Recall@k | Context Precision | 600 | 0.313 | <0.001 |
