import sonic
import bigram
import numpy
import re

text = sonic.OpenFile(r"E:\Aoban Selenia\Aoban Sonic\all.txt")
vc, z = sonic.Tokenize(text, method="num")

model = sonic.LoadTensor.Bigram(filename="sonic_model.safetensors", method="cuda")

model.tdim = round(model.dim / model.ddiv)
temp = bigram.InitModel(model)

def Generate(words, input, model_obj):
    output = []
    try:
        next_word = vc[input]
    except:
        print("Word not in corpus")
        return

    for _ in range(model_obj.mdiv + 1):
        pos_probs = {idx: 0.0 for idx in range(len(vc))}
        found = False

        for position, word in enumerate(words[:-1]):
            if word == next_word:
                following = words[position + 1]
                pos_probs[following] += 1.0
                found = True

        if not found:
            break

        keys = list(pos_probs.keys())

        p = model.forward(probs=pos_probs, model=temp, method="cuda")

        valid_mask = numpy.array([1.0 if pos_probs[k] > 0 else 0.0 for k in keys])
        p = p.ravel() * valid_mask
        total = p.sum()
        if total > 0 and not numpy.isnan(total):
            p = p / total
        else:
            p = numpy.array([pos_probs[k] for k in keys], dtype="float64")
            p = p / p.sum()

        next_word = int(numpy.random.choice(keys, p=p))
        
        output.append(next_word)
        
    return output

words = Generate(z, "the", temp)
vctrue = {value: key for key, value in vc.items()}

out = []
if words:
    for i in words: 
        print(vctrue[i])
        out.append(vctrue[i])

if words:
    output = out
    output_string = " ".join(output)

    output_string = re.sub(
            r"User:|Assistant:|[\[\]\'\",]",
            "",
            output_string
        )

    print(re.sub(r"\s+", " ", output_string).strip())
