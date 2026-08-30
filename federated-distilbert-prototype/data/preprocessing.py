import re
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"(?<!\w)@\w+")

def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = URL_RE.sub("<URL>", text)
    text = USER_RE.sub("<USER>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_csv(path: str, labels: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a text column")
    missing = [x for x in labels if x not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df
