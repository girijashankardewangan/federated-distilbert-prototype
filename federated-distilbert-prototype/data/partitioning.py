import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, train_fraction=0.70, validation_fraction=0.15, seed=42):
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")
    train, temp = train_test_split(
        df,
        test_size=1 - train_fraction,
        random_state=seed,
        shuffle=True,
    )
    relative_test = (1 - train_fraction - validation_fraction) / (1 - train_fraction)
    valid, test = train_test_split(
        temp,
        test_size=relative_test,
        random_state=seed,
        shuffle=True,
    )
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)

def dirichlet_partition(df, labels, clients, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    y = df[labels].to_numpy()
    buckets = [[] for _ in range(clients)]

    for label_idx in range(len(labels)):
        indices = np.flatnonzero(y[:, label_idx] > 0)
        if len(indices) == 0:
            continue
        proportions = rng.dirichlet(np.full(clients, alpha))
        counts = rng.multinomial(len(indices), proportions)
        start = 0
        for client_id, count in enumerate(counts):
            buckets[client_id].extend(indices[start:start + count].tolist())
            start += count

    clients_out = []
    used = set()
    for bucket in buckets:
        unique = [i for i in dict.fromkeys(bucket) if i not in used]
        used.update(unique)
        clients_out.append(df.iloc[unique].reset_index(drop=True))

    remaining = [i for i in range(len(df)) if i not in used]
    for i, idx in enumerate(remaining):
        clients_out[i % clients].loc[len(clients_out[i % clients])] = df.iloc[idx]

    return clients_out
