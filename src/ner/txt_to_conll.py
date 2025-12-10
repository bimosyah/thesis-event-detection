import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.utils import load_config


def convert_txt_to_conll(infile, outfile):
    lines = open(infile, "r", encoding="utf-8").read().splitlines()
    with open(outfile, "w", encoding="utf-8") as fw:
        for line in lines:
            if line.strip() == "":
                fw.write("\n")
                continue
            parts = line.split()
            token = parts[0]
            label = parts[1]
            fw.write(f"{token} {label}\n")


if __name__ == "__main__":
    cfg = load_config()
    convert_txt_to_conll(
        cfg["ner"]["raw_dataset"],
        cfg["ner"]["full_conll"]
    )
    print("DONE.")
