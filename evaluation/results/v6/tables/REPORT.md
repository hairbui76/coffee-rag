# System table (overall)

| System | Intent | Runs | Context Precision | Context Recall (Ragas) | Faithfulness | Answer Relevancy | Hit@k | Recall@k | Precision@k | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| MiniLM | all | 3 | 0.489 ± 0.006 | 0.191 ± 0.001 | 0.822 ± 0.002 | 0.757 ± 0.002 | 0.367 ± 0.000 | 0.269 ± 0.000 | 0.091 ± 0.000 | 0.289 ± 0.000 |
| MiniLM+RRF | all | 3 | 0.611 ± 0.002 | 0.284 ± 0.001 | 0.850 ± 0.002 | 0.787 ± 0.004 | 0.670 ± 0.000 | 0.480 ± 0.000 | 0.235 ± 0.000 | 0.548 ± 0.000 |
| BGE-M3 | all | 2 | 0.531 ± 0.002 | 0.227 ± 0.003 | 0.835 ± 0.005 | 0.808 ± 0.000 | 0.432 ± 0.000 | 0.327 ± 0.000 | 0.111 ± 0.000 | 0.342 ± 0.000 |
| BGE-M3+RRF | all | 2 | 0.634 ± 0.004 | 0.306 ± 0.002 | 0.853 ± 0.006 | 0.803 ± 0.009 | 0.672 ± 0.000 | 0.485 ± 0.000 | 0.235 ± 0.000 | 0.559 ± 0.000 |

# System table (by intent)

| System | Intent | Runs | Context Precision | Context Recall (Ragas) | Faithfulness | Answer Relevancy | Hit@k | Recall@k | Precision@k | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| MiniLM | product_search | 3 | 0.036 ± 0.000 | 0.016 ± 0.001 | 0.826 ± 0.007 | 0.859 ± 0.002 | 0.155 ± 0.000 | 0.045 ± 0.000 | 0.042 ± 0.000 | 0.103 ± 0.000 |
| MiniLM | similar_search | 3 | 0.820 ± 0.014 | 0.026 ± 0.000 | 0.823 ± 0.004 | 0.810 ± 0.005 | 0.170 ± 0.000 | 0.045 ± 0.000 | 0.039 ± 0.000 | 0.110 ± 0.000 |
| MiniLM | news_search | 3 | 0.611 ± 0.004 | 0.530 ± 0.003 | 0.818 ± 0.002 | 0.603 ± 0.011 | 0.775 ± 0.000 | 0.717 ± 0.000 | 0.193 ± 0.000 | 0.655 ± 0.000 |
| MiniLM+RRF | product_search | 3 | 0.328 ± 0.001 | 0.254 ± 0.004 | 0.890 ± 0.001 | 0.866 ± 0.002 | 0.905 ± 0.000 | 0.523 ± 0.000 | 0.439 ± 0.000 | 0.709 ± 0.000 |
| MiniLM+RRF | similar_search | 3 | 0.793 ± 0.002 | 0.021 ± 0.001 | 0.821 ± 0.004 | 0.786 ± 0.007 | 0.155 ± 0.000 | 0.040 ± 0.000 | 0.034 ± 0.000 | 0.102 ± 0.000 |
| MiniLM+RRF | news_search | 3 | 0.712 ± 0.004 | 0.578 ± 0.005 | 0.837 ± 0.003 | 0.708 ± 0.014 | 0.950 ± 0.000 | 0.879 ± 0.000 | 0.232 ± 0.000 | 0.833 ± 0.000 |
| BGE-M3 | product_search | 2 | 0.017 ± 0.001 | 0.007 ± 0.000 | 0.841 ± 0.001 | 0.861 ± 0.000 | 0.180 ± 0.000 | 0.044 ± 0.000 | 0.039 ± 0.000 | 0.090 ± 0.000 |
| BGE-M3 | similar_search | 2 | 0.829 ± 0.019 | 0.035 ± 0.002 | 0.840 ± 0.004 | 0.824 ± 0.006 | 0.185 ± 0.000 | 0.051 ± 0.000 | 0.043 ± 0.000 | 0.118 ± 0.000 |
| BGE-M3 | news_search | 2 | 0.747 ± 0.011 | 0.638 ± 0.010 | 0.826 ± 0.010 | 0.739 ± 0.004 | 0.930 ± 0.000 | 0.887 ± 0.000 | 0.252 ± 0.000 | 0.820 ± 0.000 |
| BGE-M3+RRF | product_search | 2 | 0.316 ± 0.003 | 0.247 ± 0.000 | 0.885 ± 0.005 | 0.869 ± 0.001 | 0.880 ± 0.000 | 0.491 ± 0.000 | 0.412 ± 0.000 | 0.678 ± 0.000 |
| BGE-M3+RRF | similar_search | 2 | 0.823 ± 0.018 | 0.023 ± 0.000 | 0.835 ± 0.008 | 0.812 ± 0.016 | 0.170 ± 0.000 | 0.047 ± 0.000 | 0.039 ± 0.000 | 0.116 ± 0.000 |
| BGE-M3+RRF | news_search | 2 | 0.763 ± 0.004 | 0.648 ± 0.005 | 0.838 ± 0.004 | 0.729 ± 0.011 | 0.965 ± 0.000 | 0.917 ± 0.000 | 0.254 ± 0.000 | 0.884 ± 0.000 |

# Paired tests vs BGE-M3 (per-case means across runs, bootstrap 95% CI, Wilcoxon signed-rank)

| System | Metric | n | Δ mean | 95% CI | Wilcoxon p | W/L/T |
|---|---|---|---|---|---|---|
| MiniLM vs BGE-M3 | Context Precision | 600 | -0.042 | [-0.065, -0.019] | 0.002 | 86/117/397 |
| MiniLM vs BGE-M3 | Context Recall (Ragas) | 600 | -0.036 | [-0.055, -0.017] | <0.001 | 38/62/500 |
| MiniLM vs BGE-M3 | Faithfulness | 591 | -0.012 | [-0.023, -0.001] | 0.011 | 270/318/3 |
| MiniLM vs BGE-M3 | Answer Relevancy | 600 | -0.051 | [-0.067, -0.035] | <0.001 | 249/338/13 |
| MiniLM vs BGE-M3 | Hit@k | 600 | -0.065 | [-0.100, -0.032] | <0.001 | 35/74/491 |
| MiniLM vs BGE-M3 | Recall@k | 600 | -0.059 | [-0.080, -0.036] | <0.001 | 66/93/441 |
| MiniLM vs BGE-M3 | Precision@k | 600 | -0.020 | [-0.030, -0.010] | 0.162 | 201/93/306 |
| MiniLM vs BGE-M3 | MRR | 600 | -0.053 | [-0.080, -0.026] | <0.001 | 58/110/432 |
| MiniLM+RRF vs BGE-M3 | Context Precision | 600 | +0.080 | [+0.050, +0.112] | <0.001 | 198/104/298 |
| MiniLM+RRF vs BGE-M3 | Context Recall (Ragas) | 600 | +0.058 | [+0.033, +0.082] | <0.001 | 129/56/415 |
| MiniLM+RRF vs BGE-M3 | Faithfulness | 590 | +0.016 | [+0.006, +0.026] | 0.013 | 312/276/2 |
| MiniLM+RRF vs BGE-M3 | Answer Relevancy | 600 | -0.022 | [-0.035, -0.009] | 0.096 | 279/314/7 |
| MiniLM+RRF vs BGE-M3 | Hit@k | 600 | +0.238 | [+0.197, +0.278] | <0.001 | 169/26/405 |
| MiniLM+RRF vs BGE-M3 | Recall@k | 600 | +0.153 | [+0.125, +0.182] | <0.001 | 214/48/338 |
| MiniLM+RRF vs BGE-M3 | Precision@k | 600 | +0.124 | [+0.102, +0.146] | <0.001 | 376/48/176 |
| MiniLM+RRF vs BGE-M3 | MRR | 600 | +0.205 | [+0.170, +0.242] | <0.001 | 218/60/322 |
| BGE-M3+RRF vs BGE-M3 | Context Precision | 600 | +0.103 | [+0.077, +0.129] | <0.001 | 180/74/346 |
| BGE-M3+RRF vs BGE-M3 | Context Recall (Ragas) | 600 | +0.079 | [+0.059, +0.099] | <0.001 | 117/31/452 |
| BGE-M3+RRF vs BGE-M3 | Faithfulness | 581 | +0.017 | [+0.008, +0.027] | <0.001 | 328/246/7 |
| BGE-M3+RRF vs BGE-M3 | Answer Relevancy | 600 | -0.005 | [-0.016, +0.006] | 0.435 | 309/277/14 |
| BGE-M3+RRF vs BGE-M3 | Hit@k | 600 | +0.240 | [+0.203, +0.275] | <0.001 | 154/10/436 |
| BGE-M3+RRF vs BGE-M3 | Recall@k | 600 | +0.158 | [+0.133, +0.183] | <0.001 | 189/20/391 |
| BGE-M3+RRF vs BGE-M3 | Precision@k | 600 | +0.124 | [+0.104, +0.143] | <0.001 | 189/20/391 |
| BGE-M3+RRF vs BGE-M3 | MRR | 600 | +0.217 | [+0.187, +0.246] | <0.001 | 204/23/373 |

# Metric table (one system, several recall definitions)

System: BGE-M3+RRF

| Intent | Context Recall (Ragas) | Soft Recall@k | Recall@k | Hit@k | Context Precision |
|---|---|---|---|---|---|
| product_search | 0.247 ± 0.000 | 0.507 ± 0.000 | 0.491 ± 0.000 | 0.880 ± 0.000 | 0.316 ± 0.003 |
| similar_search | 0.023 ± 0.000 | 0.086 ± 0.000 | 0.047 ± 0.000 | 0.170 ± 0.000 | 0.823 ± 0.018 |
| news_search | 0.648 ± 0.005 | 0.920 ± 0.000 | 0.917 ± 0.000 | 0.965 ± 0.000 | 0.763 ± 0.004 |
| all | 0.306 ± 0.002 | 0.504 ± 0.000 | 0.485 ± 0.000 | 0.672 ± 0.000 | 0.634 ± 0.004 |

Spearman ρ between metrics across cases:

| A | B | n | ρ | p |
|---|---|---|---|---|
| Context Recall (Ragas) | Soft Recall@k | 600 | 0.731 | <0.001 |
| Context Recall (Ragas) | Recall@k | 600 | 0.743 | <0.001 |
| Context Recall (Ragas) | Hit@k | 600 | 0.565 | <0.001 |
| Context Recall (Ragas) | Context Precision | 600 | 0.411 | <0.001 |
| Soft Recall@k | Recall@k | 600 | 0.981 | <0.001 |
| Soft Recall@k | Hit@k | 600 | 0.828 | <0.001 |
| Soft Recall@k | Context Precision | 600 | 0.181 | <0.001 |
| Recall@k | Hit@k | 600 | 0.845 | <0.001 |
| Recall@k | Context Precision | 600 | 0.175 | <0.001 |
| Hit@k | Context Precision | 600 | -0.059 | 0.146 |
