# Knowledge Distillation for Large Language Models

## Overview

Knowledge Distillation (KD) is a compression technique where a smaller model (student) is trained to replicate the behavior of a larger model (teacher).

It is commonly used after:
- Structured pruning
- Architecture reduction
- Model size scaling

KD does not increase model capacity. It transfers behavior.

---

# 1. Types of Distillation

## 1.1 Logit Distillation (Hinton, 2015)

The student matches teacher output probabilities using KL divergence.

Loss:

    L = KL(softmax(z_T / T), softmax(z_S / T)) * T²

Where:
- z_T = teacher logits
- z_S = student logits
- T = temperature (typically 2–4)

Purpose:
- Match probability distribution
- Restore calibration
- Recover fluency after pruning

---

## 1.2 Combined KL + Cross Entropy

Modern LLM training uses:

    L = α * KL + (1 - α) * CE

Why?
- KL transfers teacher behavior
- CE preserves correctness on real labels

Recommended:
- α = 0.5 to 0.8
- T = 2–4

---

## 1.3 Hidden State Matching

Student intermediate activations are matched to teacher.

Loss:

    L_hidden = MSE(h_T, h_S)

Purpose:
- Preserve internal representation geometry
- Stabilize heavy pruning
- Improve reasoning retention

Used when:
- Width pruning >30%
- Depth pruning >20%

---

# 2. Distillation Without Pruning

Distillation can be applied even if the architecture is unchanged.

Uses:
- Improve calibration
- Improve robustness
- Self-distillation

It does NOT increase capacity.

---

# 3. Distillation With Smaller Student

Teacher: 1B  
Student: 500M  

Loss:

    L = λ1 * KL + λ2 * Hidden + λ3 * CE

This transfers knowledge to a smaller architecture.

---

# 4. When Dataset Is Available

Best practice:

- Use original dataset
- Use KL + CE
- Use cosine learning rate
- Use 3–5 epochs
- Use mixed precision
- Freeze teacher

---

# 5. When Dataset Is NOT Available

Options:

## 5.1 Synthetic Data Generation

1. Sample prompts
2. Generate teacher outputs
3. Train student on teacher logits

## 5.2 Public Proxy Dataset

Use similar domain dataset and distill teacher outputs on it.

## 5.3 API Query Distillation

Query teacher model at scale and build dataset.

---

# 6. Practical Training Settings

Recommended:

- Temperature: 2–4
- Learning rate: 2e-5 to 5e-5
- Warmup: 3–5%
- Mixed precision
- Freeze teacher
- Evaluate perplexity

---

# 7. Important Limitations

Distillation:
- Cannot restore removed capacity
- Cannot create new knowledge
- Preserves behavior only

Heavy pruning requires hidden matching.

---

# 8. Production Recommendation

For stable compression:

- Attention pruning ≤25%
- MLP pruning ≤30%
- Depth pruning ≤20%
- Distill for 3 epochs minimum