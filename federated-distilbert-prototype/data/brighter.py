from datasets import load_dataset
import pandas as pd


LABELS = [
    "joy",
    "anger",
    "fear",
    "sadness",
    "surprise",
    "disgust",
]


LANGUAGE_CONFIGS = {
    "afr": "Afrikaans",
    "arq": "Algerian Arabic",
    "ary": "Moroccan Arabic",
    "chn": "Mandarin Chinese",
    "deu": "German",
    "eng": "English",
    "esp": "Spanish",
    "hau": "Hausa",
    "hin": "Hindi",
    "ibo": "Igbo",
    "ind": "Indonesian",
    "jav": "Javanese",
    "kin": "Kinyarwanda",
    "mar": "Marathi",
    "pcm": "Nigerian Pidgin",
    "ptbr": "Portuguese Brazil",
    "ptmz": "Portuguese Mozambique",
    "ron": "Romanian",
    "rus": "Russian",
    "sun": "Sundanese",
    "swa": "Swahili",
    "swe": "Swedish",
    "tat": "Tatar",
    "ukr": "Ukrainian",
    "vmw": "Makhuwa",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "zul": "Zulu",
}


DATASET_ID = "brighter-dataset/BRIGHTER-emotion-categories"


def available_languages():
    return LANGUAGE_CONFIGS.copy()


def _clean_label(value):
    if pd.isna(value):
        return 0.0

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        value = float(value)

        if pd.isna(value):
            return 0.0

        return value

    text = str(value).strip().lower()

    if text in {"", "nan", "none", "null"}:
        return 0.0

    if text in {"1", "true", "yes"}:
        return 1.0

    if text in {"0", "false", "no"}:
        return 0.0

    try:
        number = float(text)

        if pd.isna(number):
            return 0.0

        return number

    except ValueError:
        return 0.0


def _to_frame(dataset):
    frame = dataset.to_pandas()

    required_columns = ["id", "text"] + LABELS

    missing_columns = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            f"BRIGHTER dataset is missing required columns: "
            f"{missing_columns}"
        )

    frame = frame[required_columns].copy()

    frame["text"] = (
        frame["text"]
        .fillna("")
        .astype(str)
    )

    for label in LABELS:
        frame[label] = frame[label].map(_clean_label)

    frame[LABELS] = (
        frame[LABELS]
        .fillna(0.0)
        .astype("float32")
    )

    frame = frame[
        frame["text"].str.strip().ne("")
    ].reset_index(drop=True)

    if frame.empty:
        raise ValueError(
            "No usable text records were found in the dataset."
        )

    if frame[LABELS].isna().any().any():
        raise ValueError(
            "Invalid label values remain after dataset cleaning."
        )

    return frame


def load_language(config, split=None):
    if config not in LANGUAGE_CONFIGS:
        raise ValueError(
            f"Unknown BRIGHTER configuration: {config}"
        )

    if split:
        dataset = load_dataset(
            DATASET_ID,
            config,
            split=split,
        )

        return _to_frame(dataset)

    dataset = load_dataset(
        DATASET_ID,
        config,
    )

    return {
        name: _to_frame(value)
        for name, value in dataset.items()
    }


def load_languages(configs, split=None):
    loaded = [
        load_language(
            config,
            split=split,
        )
        for config in configs
    ]

    if split:
        return pd.concat(
            loaded,
            ignore_index=True,
        )

    merged = {}

    for split_name in [
        "train",
        "dev",
        "test",
    ]:
        frames = [
            item[split_name]
            for item in loaded
            if split_name in item
        ]

        if not frames:
            raise ValueError(
                f"No '{split_name}' split was found."
            )

        merged[split_name] = pd.concat(
            frames,
            ignore_index=True,
        )

    return merged


def load_all_languages(split=None):
    return load_languages(
        list(LANGUAGE_CONFIGS),
        split=split,
    )


def combine_official_splits(configs):
    train = load_languages(
        configs,
        split="train",
    )

    dev = load_languages(
        configs,
        split="dev",
    )

    test = load_languages(
        configs,
        split="test",
    )

    return train, dev, test


def build_custom_split(
    configs,
    seed=42,
    train_fraction=0.70,
    validation_fraction=0.15,
):
    from data.partitioning import split_data

    full = load_languages(
        configs,
        split=None,
    )

    frames = [
        full[split]
        for split in [
            "train",
            "dev",
            "test",
        ]
        if split in full
    ]

    if not frames:
        raise ValueError(
            "No BRIGHTER data was loaded."
        )

    merged = pd.concat(
        frames,
        ignore_index=True,
    )

    merged = (
        merged
        .drop_duplicates(
            subset=["id"]
        )
        .reset_index(drop=True)
    )

    for label in LABELS:
        merged[label] = (
            pd.to_numeric(
                merged[label],
                errors="coerce",
            )
            .fillna(0.0)
            .astype("float32")
        )

    merged["text"] = (
        merged["text"]
        .fillna("")
        .astype(str)
    )

    merged = merged[
        merged["text"].str.strip().ne("")
    ].reset_index(drop=True)

    if merged.empty:
        raise ValueError(
            "No usable records remain after BRIGHTER cleaning."
        )

    if merged[LABELS].isna().any().any():
        raise ValueError(
            "NaN values remain in BRIGHTER labels."
        )

    return split_data(
        merged,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )