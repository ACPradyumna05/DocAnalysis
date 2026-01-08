import os
import json
import glob
import random
import zipfile
from datasets import Dataset

JSON_ZIP = "peakrx.zip"       # If exists, extracts automatically
JSON_FOLDER = "./json"        # Folder alternative
OUTPUT_DATASET_PATH = "./policy_dataset"
ENCODING = "windows-1252"

# ---------------------------------------------------------
# Extract ZIP if present
# ---------------------------------------------------------
if os.path.exists(JSON_ZIP):
    print(f"Found ZIP: {JSON_ZIP} — extracting...")
    with zipfile.ZipFile(JSON_ZIP, 'r') as z:
        z.extractall("./")
    print("Extraction complete.")

if not os.path.isdir(JSON_FOLDER):
    raise RuntimeError(
        f"JSON folder '{JSON_FOLDER}' not found. "
        "Either unzip peakrx.zip or create ./json/"
    )

json_files = sorted(glob.glob(os.path.join(JSON_FOLDER, "*.json")))
print("Found JSON files:", json_files)

# ---------------------------------------------------------
# Helper functions (same logic as Colab)
# ---------------------------------------------------------
def flatten_value_to_text(val, indent=0):
    prefix = "  " * indent
    if val is None:
        return ""
    if isinstance(val, str):
        return "\n".join(prefix + ln.rstrip() for ln in val.splitlines() if ln.strip())
    if isinstance(val, (int, float, bool)):
        return prefix + str(val)
    if isinstance(val, dict):
        lines = []
        if isinstance(val.get("heading"), str):
            lines.append(prefix + val["heading"])
        for k, v in val.items():
            if k == "heading":
                continue
            if isinstance(v, str) and k.lower() == "content":
                lines.extend(flatten_value_to_text(v, indent+1).splitlines())
            else:
                if isinstance(v, (str, int, float, bool)) or v is None:
                    if isinstance(v, str) and "\n" in v:
                        lines.append(prefix + f"{k}:")
                        lines.extend(flatten_value_to_text(v, indent+1).splitlines())
                    else:
                        lines.append(prefix + f"{k}: {v}")
                else:
                    lines.append(prefix + f"{k}:")
                    lines.extend(flatten_value_to_text(v, indent+1).splitlines())
        return "\n".join(lines)
    if isinstance(val, list):
        output = []
        for i, item in enumerate(val, 1):
            if isinstance(item, dict) and "heading" in item:
                output.extend(flatten_value_to_text(item, indent).splitlines())
            else:
                output.append(prefix + f"- item {i}:")
                output.extend(flatten_value_to_text(item, indent+1).splitlines())
        return "\n".join(output)
    return prefix + str(val)


def extract_heading_content_pairs(obj):
    pairs = []
    if isinstance(obj, dict):
        if "heading" in obj and "content" in obj:
            pairs.append((obj["heading"], obj["content"]))
        for v in obj.values():
            pairs.extend(extract_heading_content_pairs(v))
    elif isinstance(obj, list):
        for item in obj:
            pairs.extend(extract_heading_content_pairs(item))
    return pairs

# ---------------------------------------------------------
# Build dataset
# ---------------------------------------------------------
sample_queries = [
    "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
    "30-year-old female, maternity cover, 1-year-old policy",
    "65-year-old male, heart surgery in Delhi, 2-year-old policy",
    "Child, dental treatment, new policy under 6 months"
]

all_examples = []

for fpath in json_files:
    with open(fpath, "r", encoding=ENCODING, errors="replace") as f:
        rawtext = f.read()

    try:
        parsed = json.loads(rawtext)
    except:
        continue

    pairs = extract_heading_content_pairs(parsed)

    if pairs:
        for (heading, content_obj) in pairs:
            clause = flatten_value_to_text(content_obj).strip()
            for query in random.sample(sample_queries, 2):
                all_examples.append({
                    "instruction": f"Given the policy clause below, answer the query: '{query}'. Return decision, amount, and justification in JSON.",
                    "input": clause,
                    "output": "{\n  \"decision\": \"<approved/rejected>\",\n  \"amount\": <number>,\n  \"justification\": \"<reference clause>\"\n}"
                })
    else:
        clause = flatten_value_to_text(parsed).strip()
        for query in random.sample(sample_queries, 2):
            all_examples.append({
                "instruction": f"Given the policy clause below, answer the query: '{query}'. Return decision, amount, and justification in JSON.",
                "input": clause,
                "output": "{\n  \"decision\": \"<approved/rejected>\",\n  \"amount\": <number>,\n  \"justification\": \"<reference clause>\"\n}"
            })

dataset = Dataset.from_list(all_examples)
dataset.save_to_disk(OUTPUT_DATASET_PATH)

print("Saved dataset:", OUTPUT_DATASET_PATH)
print("Rows:", len(dataset))
