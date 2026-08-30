import random

def typo_noise(text, rate=0.10, seed=42):
    rng = random.Random(seed)
    chars = list(text)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for i in range(len(chars)):
        if chars[i].isalpha() and rng.random() < rate:
            chars[i] = rng.choice(alphabet)
    return "".join(chars)

def remove_emojis(text):
    return "".join(ch for ch in text if ord(ch) < 128)
