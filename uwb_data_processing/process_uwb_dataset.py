import os
import re
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from utils import (
    uwb_general_corrections,
    uwb_diacritics,
    uwb_phonetic_mapping,
    uwb_number_mapping,
)

INPUT_DIR = 'UWB_Raw_Dataset'
OUTPUT_DIR = 'atc-pilot-speaker-role-classification-dataset'
MAX_WORKERS = 20

role_aliases = {'AIR': 'PILOT', 'GROUND': 'ATC', 'PILOT': 'PILOT', 'ATC': 'ATC'}
remove_tags = {
    "[noise]", "[noise_|]", "[|_noise]",
    "[speaker]", "[speaker_|]", "[|_speaker]",
    "[background_speech]", "[background_speech_|]", "[|_background_speech]",
    "[ehm_??]"
}
excluded_chars = {'+', '(', ')'}
exclusion_tags = {
    "[no_eng]", "[no_eng_|]", "[|_no_eng]",
    "[czech_|]", "[|_czech]",
    "[unintelligible]", "[unintelligible_|]", "[|_unintelligible]"
}

def extract_segments_from_trs(file_path):
    with open(file_path, 'r', encoding='cp1250') as f:
        lines = f.readlines()

    segments = []
    inside_turn = False
    buffer = []

    for line in lines:
        line = line.strip()
        if "<Turn" in line:
            inside_turn = True
            buffer = []
            continue
        if "</Turn>" in line:
            inside_turn = False
            continue
        if inside_turn:
            if line.startswith("<Sync"):
                if buffer:
                    combined = " ".join(buffer).strip()
                    if combined and combined != "..":
                        segments.append(combined)
                    buffer = []
            else:
                buffer.append(line)
    return segments

def get_roles(line):
    tags = re.findall(r'\[(.*?)\]', line)
    role_set = {
        role_aliases.get(t.upper().split('_')[0])
        for t in tags
        if t.upper().split('_')[0] in role_aliases
    }
    return {r for r in role_set if r}

def strip_inline_tags(line):
    for tag in remove_tags:
        line = line.replace(tag, "")
    return line.replace("?", "")

def strip_speaker_tags(line):
    for r in role_aliases.keys():
        line = re.sub(rf'\[(?i:({r}|{r}_\||\|_{r}))\]', '', line)
    return line

def norm(line):
    for char, replacement in uwb_diacritics.items():
        line = line.replace(char, replacement)
    line = line.replace("°", "")
    line = re.sub(r'\s+', ' ', line)
    return line.strip().upper()

def convert_atc(text):
    def repl(m):
        return uwb_phonetic_mapping.get(m.group(1).upper(), m.group(1).upper())

    text = re.sub(r'(?<=\s)([b-hj-zB-HJ-Z])(?=\s)', repl, f' {text} ').strip()

    for k, v in uwb_general_corrections.items():
        text = re.sub(rf'\b{re.escape(k)}\b', v, text, flags=re.IGNORECASE)

    tokens = text.split()
    out = []
    i = 0
    nato = set(uwb_phonetic_mapping.values())
    triggers = {
        'CONFIRM', 'REQUEST', 'DESCEND', 'CLIMB', 'MAINTAIN', 'CLEARED', 'CONTACT',
        'REPORT', 'BACKTRACK', 'LINE', 'UP', 'DOWN', 'IDENT', 'SQUAWK', 'COPY',
        'ROGER', 'WILCO', 'ACKNOWLEDGE', 'DEPART', 'APPROACH',
    }

    while i < len(tokens):
        t = tokens[i]
        if (
            len(t) == 1 and t.isalpha() and t.isupper()
            and i + 2 < len(tokens)
            and tokens[i + 1].upper() == 'AND'
            and len(tokens[i + 2]) == 1 and tokens[i + 2].isalpha() and tokens[i + 2].isupper()
        ):
            out.extend([
                uwb_phonetic_mapping[t],
                'AND',
                uwb_phonetic_mapping[tokens[i + 2]] if tokens[i + 2] != 'I' else 'I'
            ])
            i += 3
            continue
        if (
            t.upper() == 'AND'
            and i + 1 < len(tokens)
            and len(tokens[i + 1]) == 1
            and tokens[i + 1].isalpha()
            and tokens[i + 1].isupper()
        ):
            out.extend([
                'AND',
                uwb_phonetic_mapping[tokens[i + 1]] if tokens[i + 1] != 'I' else 'I'
            ])
            i += 2
            continue
        if t == 'FL':
            out.append('FLIGHT LEVEL')
        elif t in uwb_number_mapping:
            out.append(uwb_number_mapping[t])
        elif len(t) == 1 and t.isalpha() and t.isupper():
            nxt = (
                i < len(tokens) - 1 and (
                    (len(tokens[i + 1]) == 1 and tokens[i + 1].isalpha() and tokens[i + 1].isupper())
                    or tokens[i + 1].isdigit()
                    or tokens[i + 1] in nato
                    or tokens[i + 1] in triggers
                )
            )
            prev = (
                i > 0 and (
                    (len(tokens[i - 1]) == 1 and tokens[i - 1].isalpha() and tokens[i - 1].isupper())
                    or tokens[i - 1].isdigit()
                    or tokens[i - 1] in nato
                )
            )
            if i == 0:
                out.append(
                    uwb_phonetic_mapping[t] if nxt else (t if t == 'I' else uwb_phonetic_mapping.get(t, t))
                )
            else:
                out.append(
                    uwb_phonetic_mapping[t] if prev or nxt else t
                )
        else:
            out.append(t)
        i += 1

    if out and len(out[-1]) == 1 and out[-1].isalpha() and out[-1].isupper():
        out[-1] = uwb_phonetic_mapping[out[-1]]

    return ' '.join(out)

def process_file(file_path):
    local_results = []
    segments = extract_segments_from_trs(file_path)

    for s in segments:
        if any(c in s for c in excluded_chars):
            continue
        if any(t in s for t in exclusion_tags):
            continue

        roles_in_line = get_roles(s)
        if len(roles_in_line) != 1:
            continue
        role = roles_in_line.pop()

        cleaned = strip_inline_tags(s)
        cleaned = strip_speaker_tags(cleaned)
        cleaned = norm(f"[{role}] {cleaned}")

        if not cleaned[len(f"[{role}]"):].strip():
            continue

        converted = norm(convert_atc(cleaned))
        if re.search(r'\d', converted):
            continue
        if len(converted.split()) < 4:
            continue

        local_results.append((role, converted))
    return local_results

def process_dataset(input_dir, output_dir):
    random.seed(42)
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".trs")
    )
    all_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in tqdm(executor.map(process_file, files), total=len(files), desc="Processing Dataset"):
            all_results.extend(result)

    atc_lines = [line for role, line in all_results if role == 'ATC']
    pilot_lines = [line for role, line in all_results if role == 'PILOT']
    min_len = min(len(atc_lines), len(pilot_lines))
    balanced_lines = [('ATC', l) for l in atc_lines[:min_len]] + [('PILOT', l) for l in pilot_lines[:min_len]]
    random.shuffle(balanced_lines)

    role_counts = {'ATC': 1, 'PILOT': 1}
    for role, line in balanced_lines:
        index = role_counts[role]
        filename = f"{role}_{index}.txt"
        role_counts[role] += 1

        cleaned_line = re.sub(r'^\[(ATC|PILOT)\]\s*', '', line)
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            f.write(cleaned_line + '\n')

    print("UWB Pilot-ATC Classification Dataset Creation Complete.")

if __name__ == "__main__":
    process_dataset(INPUT_DIR, OUTPUT_DIR)
