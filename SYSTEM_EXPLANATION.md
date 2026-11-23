# PIGuard System Explanation

## 1. System Overview

This project reproduces the evaluation of:

> **PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free**

PIGuard is a binary text-classification model for detecting prompt injection attacks. Given an input text, the model predicts one of two classes:

- `benign`: normal input
- `injection`: prompt injection attack

PIGuard is a security classifier, not a generative chatbot.

## 2. Motivation

Traditional prompt-injection detectors may suffer from **over-defense**. Over-defense occurs when a model incorrectly blocks a benign input because it contains words or phrases commonly associated with attacks.

For example:

```text
How can I safely execute this command?
```

This may be a legitimate question even though it contains words related to commands or execution.

PIGuard proposes **Mitigating Over-defense for Free (MOF)**. The goal is to:

1. Detect genuine prompt injection attacks.
2. Reduce false positives on benign inputs.
3. Handle benign sentences containing prompt-injection trigger words more reliably.

## 3. System Architecture

The main processing flow is:

```text
Input text
    |
    v
DeBERTa-v3-base encoder
    |
    v
Text representation
    |
    v
Linear classification layer
    |
    v
benign / injection
```

The main implementation is in:

- [`PIGuard.py`](./PIGuard.py)
- [`eval.py`](./eval.py)
- [`eval_hf.py`](./eval_hf.py)

### 3.1 Text Encoding

PIGuard uses `microsoft/deberta-v3-base` as its base language model. The encoder converts the input text into a numerical representation.

### 3.2 Classification

The model uses the hidden representation of the first token and passes it through a linear layer:

```text
DeBERTa hidden representation -> Linear layer -> 2 classes
```

The two class IDs are:

```text
0: benign
1: injection
```

### 3.3 Inference

The model produces classification logits for both classes. The class with the larger logit is selected as the prediction.

## 4. Datasets

The local datasets are stored in [`datasets/`](./datasets/).

### 4.1 Training and Validation Data

- `train.json`: approximately 76,735 training samples
- `valid.json`: 144 validation samples

The evaluation described in this document does not retrain the model. It uses the publicly released checkpoint:

```text
leolee99/PIGuard
```

### 4.2 NotInject

NotInject evaluates over-defense. Its samples are benign, but contain different numbers of prompt-injection trigger words.

The local subsets are:

- `NotInject_one.json`: 113 samples
- `NotInject_two.json`: 113 samples
- `NotInject_three.json`: 113 samples

There are 339 samples in total.

This dataset measures whether the model incorrectly classifies benign, trigger-rich text as an injection.

### 4.3 BIPIA

BIPIA evaluates the detection of indirect prompt injection attacks.

The local files are:

- `BIPIA_text.json`: 75 samples
- `BIPIA_code.json`: 50 samples

The evaluation reports text and code subsets separately and then averages their accuracies.

### 4.4 WildGuard-Benign

The local file is:

- `wildguard.json`: 971 benign samples

This dataset measures the model's ability to correctly recognize benign inputs.

### 4.5 PINT Limitation

PINT is part of the paper's complete evaluation protocol. However, the full PINT benchmark requires access from the dataset authors.

The local `PINT.json` is not suitable for the complete evaluation because:

1. It contains only eight records.
2. Its category names do not match the categories expected by the official evaluation code.

Therefore, PINT was not included in the reported results. No samples were fabricated and no substitute dataset was silently used.

## 5. Running the Evaluation

Open PowerShell and move to the repository:

```powershell
cd "D:\新建文件夹\School\毕业设计\PIGuard"
```

Run the evaluation with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -u .\run_partial_eval.py
```

The script is [`run_partial_eval.py`](./run_partial_eval.py). It:

1. Loads `leolee99/PIGuard`.
2. Loads the local NotInject, BIPIA, and WildGuard-Benign datasets.
3. Classifies every sample.
4. Compares predictions with the expected labels.
5. Calculates accuracy.
6. Saves the results as JSON.

The first run may download the model from Hugging Face. Later runs can reuse the local model cache.

To save the console output:

```powershell
.\.venv\Scripts\python.exe -u .\run_partial_eval.py *> .\logs\partial_metrics_manual.txt
```

The structured results are saved to:

- [`logs/partial_metrics_manual.json`](./logs/partial_metrics_manual.json)
- [`logs/partial_metrics_manual.txt`](./logs/partial_metrics_manual.txt)

## 6. Evaluation Results

The experiment was run with:

- Model: `leolee99/PIGuard`
- Device: CPU
- Evaluation scope: NotInject, BIPIA, and WildGuard-Benign
- Exit code: `0`

| Dataset | Metric | Result |
|---|---|---:|
| NotInject-one | Accuracy | 94.69% |
| NotInject-two | Accuracy | 89.38% |
| NotInject-three | Accuracy | 81.42% |
| **NotInject overall** | Average accuracy | **88.50%** |
| BIPIA-text | Injection accuracy | 38.67% |
| BIPIA-code | Injection accuracy | 98.00% |
| **BIPIA overall** | Average accuracy | **68.33%** |
| WildGuard-Benign | Benign accuracy | **76.11%** |

The repository calculates the aggregate metrics as:

```text
NotInject overall = average(NotInject-one, NotInject-two, NotInject-three)
BIPIA overall = average(BIPIA-text, BIPIA-code)
```

## 7. Interpretation of the Results

### 7.1 NotInject

The NotInject results are:

```text
One trigger:    94.69%
Two triggers:   89.38%
Three triggers: 81.42%
```

The accuracy decreases as the number of trigger words increases. This suggests that the model reduces over-defense compared with models that rely heavily on trigger words, but it remains affected by the number of trigger words in a benign sentence.

### 7.2 BIPIA

BIPIA-code reaches 98.00%, while BIPIA-text reaches 38.67%. This shows that performance varies substantially between attack formats. The overall BIPIA score should therefore not be interpreted as equally strong performance on both text and code attacks.

The BIPIA overall value is the simple average of the two subset accuracies:

```text
(38.67% + 98.00%) / 2 = 68.33%
```

### 7.3 WildGuard-Benign

The WildGuard-Benign accuracy is 76.11%. This indicates that the model correctly recognizes most, but not all, of the benign samples in this local evaluation set.

## 8. Reproduction Status

This experiment is a **partial reproduction**, not a complete reproduction of every paper experiment.

Completed:

- Official released PIGuard checkpoint loaded successfully.
- NotInject evaluation completed.
- BIPIA evaluation completed.
- WildGuard-Benign evaluation completed.
- Raw output and structured JSON results saved.

Not completed:

- Full PINT evaluation.
- Training a new PIGuard checkpoint from scratch.

The full PINT evaluation requires obtaining the official benchmark through the access procedure documented by the repository authors.

## 9. Questions a Teacher May Ask

### Did you retrain the model?

No. This experiment evaluates the publicly released `leolee99/PIGuard` checkpoint. The purpose was to reproduce the evaluation procedure first. Re-training would require matching the original training data, random seed, software versions, hardware, and checkpoint-selection procedure.

### Why was PINT not evaluated?

The complete PINT benchmark requires access from its authors. The local file named `PINT.json` contains only eight records and does not use the categories expected by the official evaluator. It was therefore excluded rather than fabricated or replaced.

### Why might the results differ from the paper?

Possible causes include:

- Different model or dataset versions.
- Differences in preprocessing.
- Different Python, PyTorch, or Transformers versions.
- Different hardware.
- Differences between the locally available data and the exact data used in the original experiment.

The reported values are reproducible results for the current local environment and should not be presented as an exact reproduction of the authors' original environment.

### Why is accuracy used?

The official evaluation code uses accuracy:

- For NotInject, accuracy is the proportion of benign samples correctly classified as benign.
- For BIPIA, accuracy is the proportion of injection samples correctly classified as injection.
- For WildGuard-Benign, accuracy is the proportion of benign samples correctly classified as benign.

### How is a prediction judged as correct?

For each sample, the model predicts either `benign` or `injection`. The predicted label is compared with the expected label in the dataset. Accuracy is calculated as:

```text
accuracy = correctly classified samples / total samples
```

## 10. Suggested Presentation Summary

> This project reproduces the evaluation of PIGuard, a DeBERTa-v3-base binary classifier for detecting prompt injection attacks. The main research problem is over-defense, where benign text containing attack-related trigger words is incorrectly blocked. PIGuard uses the MOF strategy to reduce this problem.
>
> I used the authors' publicly released `leolee99/PIGuard` checkpoint and ran the official evaluation logic on the locally available NotInject, BIPIA, and WildGuard-Benign datasets. The results were 88.50% on NotInject, 68.33% on BIPIA, and 76.11% on WildGuard-Benign.
>
> The NotInject accuracy decreased as the number of trigger words increased, showing that trigger words still affect the model. BIPIA-code performance was high at 98.00%, while BIPIA-text performance was lower at 38.67%, showing different generalization performance across attack formats. Because the full PINT benchmark requires access from its authors, this work should be described as a partial reproduction rather than a complete reproduction.
