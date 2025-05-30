# ATC-Pilot Speaker Role Classification Task

This repository provides a complete pipeline for a binary sequence classification task that identifies whether a given air traffic communication utterance was spoken by a **pilot** or an **air traffic controller (ATC)**, using only textual input.

Unlike traditional approaches that rely on audio or acoustic cues (such as voice characteristics or channel separation), this project addresses the task entirely in the **text domain**, leveraging transformer-based architectures to make speaker role predictions based on the lexical and structural content of each transmission.

## Task Description

The model performs binary classification on single-turn utterances, assigning one of two speaker roles:

- `PILOT`
- `ATC`

Each utterance is treated independently without relying on dialogue or turn-level context.

## Model Performance

The final fine-tuned model achieves the following metrics on the test set:

- **Accuracy**: 96.64%  
- **Precision**: 96.40%  
- **Recall**: 96.91%  
- **F1 Score**: 96.65%

## Model Architecture

- Base model: `microsoft/deberta-v3-large`
- Task: Binary Sequence Classification (`num_labels=2`)
- Key training configurations:
  - Cosine learning rate scheduler with warmup (10%)
  - Batch size: 128
  - Early stopping based on F1 score
  - Max sequence length: 256 tokens
  - Mixed-precision training (FP16)
  - Validation every 200 steps

## Intended Use

This model is suitable for:

- Speaker role tagging in air traffic communication transcripts
- Text-only preprocessing in multi-modal ATC systems
- Filtering or segmenting large corpora for downstream aviation language processing tasks

## Limitations

- The model operates on **single-turn utterances** and does not incorporate preceding or following context.
- Certain ambiguous transmissions (e.g., "ROGER", "THANK YOU") may not be attributable from text alone.
- In scenarios requiring high-confidence classification under ambiguity, acoustic features or metadata should be used in conjunction.

## Example Predictions

Use the following format to test predictions:

```  
Input: "CLEARED FOR TAKEOFF RUNWAY ONE ONE LEFT"  
Prediction: "ATC"

Input: "REQUESTING PUSHBACK"  
Prediction: "PILOT"  
```

## Benchmark Comparison

This work builds upon and improves prior text-based speaker role classification research. For example, a related model by [Juan Zuluaga-Gomez](https://huggingface.co/Jzuluaga/bert-base-speaker-role-atc-en-uwb-atcc), which uses a BERT-base architecture, achieves:

- **Accuracy**: 89.03%  
- **Precision**: 87.10%  
- **Recall**: 91.63%  
- **F1 Score**: 89.31%

In comparison, this repository presents a **DeBERTa-v3-large** model with significantly improved performance:

- **Accuracy**: 96.64%  
- **Precision**: 96.40%  
- **Recall**: 96.91%  
- **F1 Score**: 96.65%

Evaluation notebooks (`evaluate_juans_model.ipynb` and `evaluate_jacks_model.ipynb`) are provided to reproduce these comparisons using the same test set.

## Repository Structure

This repository includes all necessary tools to preprocess text data, fine-tune the model, and evaluate performance.

### Training

- **`training_script/train.py`**  
  Fine-tunes the model using a preprocessed dataset. Includes FP16 support, early stopping, and evaluation.

### Evaluation

- **`evaluation_scripts/evaluate_jacks_model.ipynb`**  
  Runs full evaluation of the DeBERTa-v3-large model with classification metrics.

- **`evaluation_scripts/evaluate_juans_model.ipynb`**  
  Compares Juan Zuluaga-Gomez’s BERT-based model on the same test set.

### Utilities

- **`utils/save_model_from_checkpoint.py`**  
  Converts training checkpoint directories into standalone Hugging Face-compatible model folders.

- **`utils/upload_model_to_hf.py`**  
  Uploads a trained model and tokenizer to the Hugging Face Hub.

- **`utils/requirements.txt`**  
  Lists all Python packages required for training and evaluation.

## References

- [Juan Zuluaga-Gomez’s Hugging Face Model](https://huggingface.co/Jzuluaga/bert-base-speaker-role-atc-en-uwb-atcc)
- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention (Microsoft)](https://github.com/microsoft/DeBERTa)