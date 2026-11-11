# Sonic

Sonic is a lightweight statistical AI model library written in Python which provides AI features for Deep Learning.
It is intended to be a lightweight and simplistic alternative to PyTorch, having easy-to-use functions.

## Libraries
Sonic consists of these lightweight components:

| Component | Description | 
| -------- | -------- |
| sonic    | General usage library   | 
| sonic.bigram | Library to make a Sonic-Native Bigram. |
| sonic.transformer    | Transformer Implemenation, work in progress     | 

## Installation

```bash
pip install sonicperformance
```

## Usage

This is an implementation of a basic Bigram Model.

```python
import sonic
import sonic.bigram
import numpy
import re 

text = sonic.OpenFile("yourfile.txt") # replace with a string or your file path
words = sonic.Tokenize(text)

model = sonic.bigram.Model(32, 16)

temp = sonic.bigram.InitModel(model)

def Generate(words, input, model):
    output = []
    next_word = input

    for _ in range(model.mdiv + 1):
        pos_probs = {}

        for position, word in enumerate(words[:-1]):
            if word == next_word:
                following = words[position + 1]
                pos_probs[following] = pos_probs.get(following, 0) + 1

        if not pos_probs:
            break

        keys = list(pos_probs.keys())

        p = sonic.p(probs=pos_probs, model=model, method="cpu")

        # or use cupy here
        next_word = str(
            numpy.random.choice(keys, p=p)
        )
        
        output.append(next_word)

    output_string = " ".join(output)

    output_string = re.sub(
        r"User:|Assistant:|[\[\]\'\",]",
        "",
        output_string
    )

    return re.sub(r"\s+", " ", output_string).strip()

print(Generate(words, "the", temp))
```

## Features

* **Text tokenization**: Tokenize text into a list, or turn them into token embeddings
* Word frequency analysis
* Text generation

## Generation Implementation
This is a baseline level implementation for generation using Sonic.

```python
def Generate(words, input, model):
    output = []
    next_word = input

    for _ in range(model.mdiv + 1):
        pos_probs = {}

        for position, word in enumerate(words[:-1]):
            if word == next_word:
                following = words[position + 1]
                pos_probs[following] = pos_probs.get(following, 0) + 1

        if not pos_probs:
            break

        keys = list(pos_probs.keys())

        p = sonic.p(probs=pos_probs, model=model, method="cpu")

        next_word = str(
            numpy.random.choice(keys, p=p)
        )
        
        output.append(next_word)

    output_string = " ".join(output)

    output_string = re.sub(
        r"User:|Assistant:|[\[\]\'\",]",
        "",
        output_string
    )

    return re.sub(r"\s+", " ", output_string).strip()
```

## Requirements
* Cupy
* NumPy

## License

MIT

## Incorporation
Sonic can be used as a pipeline in text-generation projects, tokenizing, or NLP experimenting for beginners.