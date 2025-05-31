import os
import hashlib
import random
import string
from datasets import Dataset, DatasetDict, concatenate_datasets

dataset_folder_name = "atc-pilot-speaker-role-classification-dataset"

def generate_id(text, seed=42):
    random.seed(hashlib.md5((text + str(seed)).encode()).hexdigest())
    return ''.join(random.choices(string.ascii_letters + string.digits, k=20))

def create_text_classification_dataset(folder):
    ids, classes, texts = [], [], []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if not text:
                    continue
                label = filename.split("_")[0].lower()
                sample_id = generate_id(text)
                ids.append(sample_id)
                classes.append(label)
                texts.append(text)
    return Dataset.from_dict({
        "id": ids,
        "class": classes,
        "text": texts
    })

full_dataset = create_text_classification_dataset(dataset_folder_name)
full_dataset = full_dataset.shuffle(seed=42)

pilot_dataset = full_dataset.filter(lambda example: example['class'] == 'pilot')
atc_dataset = full_dataset.filter(lambda example: example['class'] == 'atc')

min_len = min(len(pilot_dataset), len(atc_dataset))
pilot_dataset = pilot_dataset.select(range(min_len))
atc_dataset = atc_dataset.select(range(min_len))

def split_by_proportion(dataset, train_frac=0.8, val_frac=0.1, seed=42):
    test_frac = 1 - train_frac - val_frac
    split_1 = dataset.train_test_split(test_size=test_frac, seed=seed)
    temp_train = split_1["train"]
    test = split_1["test"]
    val_size = val_frac / (train_frac + val_frac)
    split_2 = temp_train.train_test_split(test_size=val_size, seed=seed)
    return split_2["train"], split_2["test"], test

pilot_train, pilot_val, pilot_test = split_by_proportion(pilot_dataset)
atc_train, atc_val, atc_test = split_by_proportion(atc_dataset)

train = concatenate_datasets([pilot_train, atc_train]).shuffle(seed=42)
validation = concatenate_datasets([pilot_val, atc_val]).shuffle(seed=42)
test = concatenate_datasets([pilot_test, atc_test]).shuffle(seed=42)

dataset_dict = DatasetDict({
    "train": train,
    "validation": validation,
    "test": test
})

dataset_dict.push_to_hub(dataset_folder_name, private=True)