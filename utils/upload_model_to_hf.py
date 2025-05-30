from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer

model_dir = "atc_pilot_speaker_role_classification_model"

model = DebertaV2ForSequenceClassification.from_pretrained(model_dir)
tokenizer = DebertaV2Tokenizer.from_pretrained(model_dir)

model.push_to_hub("atc_pilot_speaker_role_classification_model", private=True)
tokenizer.push_to_hub("atc_pilot_speaker_role_classification_model", private=True)