## BASIC MODEL (≈7.29M params)

### Timeline of Improvements

| Phase              | Key Change            | Val Loss (Start → Best) | Notes                                |
| ------------------ | --------------------- | ------------------------ | ------------------------------------ |
| Baseline           | Random init, fixed LR | 123.69 →**5.72**        | Fast convergence, early saturation   |
| Init Weights       | Better weight init    | 10.84 →**3.88**         | Massive jump; init dominates         |
| + Cosine Decay     | Scheduler added       | 10.86 →**3.79**         | Marginal gain, smoother              |
| + Packed Tokenizer | Sequence packing      | 10.84 →**2.89**         | **Best result**; data efficiency win |

### Final Metrics

| Metric            | Value                          |
| ----------------- | ------------------------------ |
| **Worst val loss**    | 123.72                         |
| **Best val loss**     | **2.89**                       |
| **Training Time** | ~5.5–8 min                    |
| **Stability**     | 🟢 Very stable (no divergence) |

## MID-LEVEL MODEL (≈12.39M params)

### Timeline of Improvements

| Phase              | Key Change               | Val Loss (Start → Best) | Notes                   |
| ------------------ | ------------------------ | ------------------------ | ----------------------- |
| Baseline           | Random init              | 186.86 →**4.90**        | Long plateau            |
| Init Weights       | Re-init                  | 10.85 →**3.50**         | Major reset improvement |
| + Cosine Decay     | LR scheduling            | 10.85 →**3.50**         | Gradual refinement      |
| + Packed Tokenizer | Better token utilization | 10.84 →**2.96**         | **Largest single gain** |

### Final Metrics

| Metric            | Value                           |
| ----------------- | ------------------------------- |
| **Worst val loss**    | 186.89                          |
| **Best val loss**     | **2.96**                        |
| **Training Time** | ~30–32 min                     |
| **Stability**     | 🟢 Stable (tight train/val gap) |

## QUALITY MODEL (≈40.73M params)

### Timeline of Improvements

| Phase              | Key Change        | Val Loss (Start → Best) | Notes                        |
| ------------------ | ----------------- | ------------------------ | ---------------------------- |
| Baseline           | Random init       | 376.42 →**4.97**        | Slow, unstable mid-training  |
| Instability        | LR too high       | ↑ to ~7.3               | Clear divergence episode     |
| Init Weights       | Reset             | 10.94 →**1.86**         | Model finally usable         |
| + Packed Tokenizer | Efficient context | 10.94 →**1.86**         | **Best overall performance** |

### Final Metrics

| Metric            | Value                        |
| ----------------- | ---------------------------- |
| **Worst val loss**    | 376.47                       |
| **Best val loss**     | **1.86**                     |
| **Training Time** | ~2.5 hours                   |
| **Stability**     | 🟢 Stable (best performance) |

## FINAL COMPARISON TABLE

| Category      | Params | Worst Loss | Best Loss | Training Time | Stability  |
| ------------- | -----: | ---------: | --------: | ------------: | ---------- |
| **Basic**     |  7.29M |     123.72 |  **2.89** |     ~6–8 min | 🟡 OKay   |
| **Mid-Level** | 12.39M |     186.89 |  **2.96** |       ~30 min | 🟢 Balance |
| **Quality**   | 40.73M |     376.47 |  **1.86** |      ~2.5 hrs | 🟢 Best    |

## What Improved Model Performance (and Why)

- **Weight Initialization**

  - Resetting from poor random baselines immediately collapsed loss by an order of magnitude.
  - Proper init places activations in a trainable regime, preventing early optimization collapse.
- **Packed Tokenization**

  - Increased token utilization per batch by reducing padding waste.
  - Directly improves effective data throughput → lower loss with the same compute.
  - This was the **single biggest contributor** across all model sizes.
- **Cosine Learning Rate Decay**

  - Smoothed late-stage optimization and prevented oscillations.
  - Helped convergence consistency, not raw performance gains.
- **Model Scale (Conditional Benefit)**

  - Larger models only helped **after** fixing data efficiency and initialization.
  - Without those, scale amplified instability rather than quality.
- **Learning Rate Control**

  - Especially critical for larger models.
  - Over-aggressive LR caused visible divergence even after long stable phases.

**Points to remember**
Performance gains came from **training mechanics and data efficiency**, not from blindly increasing parameter count.
