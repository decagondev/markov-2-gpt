# Markov Chain Text Generator

import random
from collections import defaultdict

# Sample corpus
corpus = """
Once upon a time, there was a brave princess who...
She lived in a castle far, far away.
One day, she met a dragon.
"""

# Tokenize the text
words = corpus.replace('\n', ' ').split()

# Build the transition dictionary
transitions = defaultdict(list)
for i in range(len(words) - 1):
    transitions[words[i]].append(words[i + 1])

# Function to generate text
def generate_text(start_word, length=20):
    if start_word not in transitions:
        return "Start word not in corpus."
    current_word = start_word
    output = [current_word]
    for _ in range(length - 1):
        next_words = transitions.get(current_word, None)
        if not next_words:
            break
        current_word = random.choice(next_words)
        output.append(current_word)
    return ' '.join(output)

# Example usage
start = "She"
generated = generate_text(start, 30)
print("Generated Text:")
print(generated)

# Optional: Display the transition dictionary
# for key, value in transitions.items():
#     print(f"{key}: {value}")
