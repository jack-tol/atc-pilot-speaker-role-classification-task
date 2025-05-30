from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification

checkpoint_path = "PATH_TO_BEST_CHECKPOINT"
output_path = "atc_pilot_speaker_role_classification_model"

model = DebertaV2ForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)