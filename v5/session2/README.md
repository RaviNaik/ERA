# BPE Tokenizer Assignment — Full Experiment Report

**Course**: ERA (Extensive Research in AI) — Session 2  
**Task**: Design a multilingual BPE tokenizer for India's Wikipedia pages in English, Hindi, Telugu, and Kannada with a shared vocabulary of 10,000 tokens, minimizing the fertility ratio spread across languages.  
**Scoring formula**: `Score = 1000 / (X_max - X_min)`  
where `X_i = total subword tokens produced / total faithful units` for language `i`.

---

## 1. Corpus Statistics

Data source: Wikipedia HTML converted to faithful Markdown using the Wikipedia REST API (`/api/rest_v1/page/html/`). This preserves all links, lists, headers, tables, and special characters.

| Language | Wikipedia Title | Chars | Faithful Units |
|----------|----------------|------:|------:|
| English  | India           | 601,843 | 186,367 |
| Hindi    | भारत            | 463,156 |  88,359 |
| Telugu   | భారతదేశం        | 199,575 |  36,292 |
| Kannada  | ಭಾರತ            |  64,057 |  12,293 |
| **Total**|                | **1,328,631** | **323,311** |

> **Note**: A "faithful unit" is defined by the regex `r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]"`. This counts contiguous alphanumeric runs as single units, and any visible punctuation/symbols as single units. This is the denominator for fertility ratio calculation.

---

## 2. Tokenizer Architecture

All experiments (unless otherwise noted) use the following standard architecture:
- **Pre-tokenizer**: `Metaspace` (using `▁` to preserve whitespace natively)
- **Normalizer**: `NFKC`
- **Decoder**: `Metaspace`
- **Vocab Size**: `10,000`

The use of `Metaspace` ensures strict faithfulness — `decode(encode(text)) == text` is completely preserved, including URLs, brackets, and punctuation, which was a critical evaluator requirement.

---

## 3. Experiment Progression

Our methodology involved running six progressive experiments, using insights from each to inform the next, leading to the optimal submitted weights.

### Experiment Mapping

| Script Name | Experiment | Description |
|---|---|---|
| `exp1_en_only.py` | Experiment 1 | English-Only Baseline |
| `exp2_naive.py` | Experiment 2 | Naive Multilingual (Equal Weights) |
| `exp3_oversample.py` | Experiment 3 | Uniform Indic Oversampling (×2) |
| `exp4_differential.py` | Experiment 4 | Differential Oversampling (Inverse Corpus) |
| `exp5_merged.py` | Experiment 5 | Merged Vocabulary (Independent Budgets) |
| `train_tokenizer.py` | Experiment 6 | **Focused Sweet-Spot BPE (Submitted)** |

*Note: The final `train_tokenizer.py` implements Experiment 6. The `run_all_experiments.py` orchestrator trains and evaluates all 6 experiments in sequence to generate the metrics.*

---

## 4. Final Results (Experiment 6: Focused Sweet-Spot)

**Configuration**:
- Training weights: **en×1, hi×1, te×2, kn×4**

By studying the corpus size disparities, we identified that English and Hindi are large enough to not need oversampling. Only Telugu and Kannada require targeted boosts.

**Token Statistics**:

| Language | Faithful Units | Tokens Produced | Fertility Ratio (X) |
|---|---:|---:|---:|
| English  | 186,367 |  118,098 | **0.6337** |
| Hindi    |  88,359 |   57,834 | **0.6545** |
| Telugu   |  36,292 |   23,115 | **0.6369** |
| Kannada  |  12,293 |    7,676 | **0.6244** |

**Score Calculation**:

```
X_min = 0.6244 (Kannada)
X_max = 0.6545 (Hindi)
Spread = 0.6545 - 0.6244 = 0.0301
Score = 1000 / 0.0301 = 33,207
Hindi Penalty Factor = 1.0 (since 0.6545 < 1.2)
Final Score = 33,207
```

**Conclusion**: This focused oversampling brings all four languages into an exceptionally tight cluster (spread 0.030), yielding a final score of **33,207**.

---

## 5. Web Application

An interactive web application is available to visualize the progression of all experiments, explore the corpus, and test the tokenizers.

**Live Demo**: [https://ravinaik.github.io/ERA/v5/session2/](https://ravinaik.github.io/ERA/v5/session2/)

The app uses `Chart.js` for visualizations and allows downloading the `tokenizer.json` and `metrics.json` directly.
