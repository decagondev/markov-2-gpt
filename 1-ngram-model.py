# N-gram Language Model (Trigram Example)

import random
from collections import defaultdict

# Sample corpus
corpus = """
Once upon a time, there was a brave princess who lived in a castle.
One day, the princess met a wise old dragon in the forest.
They became friends and had many adventures together.
"""

# Tokenize the corpus
words = corpus.replace('\n', ' ').split()

# Create trigram model (n=3)
n = 3
ngrams = defaultdict(list)

for i in range(len(words) - n + 1):
    key = tuple(words[i:i + n - 1])  # (w1, w2)
    next_word = words[i + n - 1]     # w3
    ngrams[key].append(next_word)

# Function to generate text
def generate_text(start_sequence, length=20):
    if tuple(start_sequence) not in ngrams:
        return "Start sequence not in corpus."
    output = list(start_sequence)
    current_seq = tuple(start_sequence)
    for _ in range(length - len(start_sequence)):
        possible_next = ngrams.get(current_seq, [])
        if not possible_next:
            break
        next_word = random.choice(possible_next)
        output.append(next_word)
        current_seq = tuple(output[-(n - 1):])  # Slide window
    return ' '.join(output)

# Example usage
start = ["the", "princess"]
print("Generated Text:")
print(generate_text(start, 30))
