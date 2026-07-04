"""One-off: list the real file structure of the gated HF repo so download_data.py
can be written against actual filenames instead of guesses."""
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

api = HfApi(token=os.environ["HF_TOKEN"])
files = api.list_repo_files("databricks/officeqa", repo_type="dataset")

print(f"{len(files)} files total\n")

top_level = sorted({f.split("/")[0] for f in files})
print("Top-level entries:", top_level)

for prefix in top_level:
    matches = [f for f in files if f.startswith(prefix + "/")]
    if matches:
        print(f"\n{prefix}/  ({len(matches)} files) sample:")
        for f in matches[:15]:
            print(" ", f)
