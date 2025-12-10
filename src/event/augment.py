import os
import random

import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

from src.common.utils import load_config, save_df, set_seed

cfg = load_config()
set_seed(42)


def mask_and_replace(text, masker, tokenizer, replace_prob=0.15, top_k=5):
    tokens = tokenizer.tokenize(text)
    n = max(1, int(len(tokens) * replace_prob))
    idxs = list(range(len(tokens)))
    random.shuffle(idxs)
    masked_idxs = sorted(idxs[:n])
    masked_text = []
    token_iter = iter(tokens)
    # rebuild text with mask tokens in place of selected token indices
    for i, t in enumerate(tokens):
        if i in masked_idxs:
            masked_text.append(tokenizer.mask_token)
        else:
            masked_text.append(t)
    masked_text_str = tokenizer.convert_tokens_to_string(masked_text)
    # predict fill
    try:
        preds = masker(masked_text_str, top_k=top_k)
    except:
        return text
    # masker returns list of dicts or nested list depending on huggingface; best to take top predictions for single mask situations.
    # For multi-mask output, the pipeline handles sequentially only for single mask. So here we fallback: replace one token at time.
    # Simpler approach: replace tokens one by one.
    out = masked_text.copy()
    for mi in masked_idxs:
        masked_piece = tokenizer.convert_tokens_to_string(out)
        try:
            res = masker(masked_piece, top_k=top_k)
            if isinstance(res, list) and len(res) > 0:
                choice = res[0]['token_str']
                # clean
                choice = choice.strip()
                out[mi] = choice
            else:
                out[mi] = tokens[mi]
        except:
            out[mi] = tokens[mi]
    return tokenizer.convert_tokens_to_string(out)


def augment_train(csv_in, csv_out, factor=2):
    df = pd.read_csv(csv_in)
    model_name = cfg["transformer"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mlm = AutoModelForMaskedLM.from_pretrained(model_name)
    masker = pipeline("fill-mask", model=mlm, tokenizer=tokenizer, device=0)

    new_rows = []
    for idx, row in df.iterrows():
        text = row["message"]
        lable = row["lable"]
        new_rows.append({"message": text, "lable": lable})
        for k in range(factor - 1):
            aug = mask_and_replace(text, masker, tokenizer, replace_prob=cfg["augmentation"]["mlm"]["replace_prob"],
                                   top_k=cfg["augmentation"]["mlm_top_k"])
            # small sanity check:
            if aug.strip() == "":
                aug = text
            new_rows.append({"message": aug, "lable": lable})
    out_df = pd.DataFrame(new_rows)
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    out_df = out_df.sample(frac=1, random_state=cfg["random_state"]).reset_index(drop=True)
    save_df(out_df, csv_out)
    print(f"Augmented saved to {csv_out} size={len(out_df)}")


if __name__ == "__main__":

    p = cfg["processed_dir"]
    train_csv = os.path.join(p, "train.csv")
    ad = cfg["augmented_dir"]
    os.makedirs(ad, exist_ok=True)
    for f in cfg["augmentation"]["augment_factors"]:
        out = os.path.join(ad, f"train_aug_x{f}.csv")
        augment_train(train_csv, out, factor=f)
