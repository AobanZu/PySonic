import os
import re
import cupy
import numpy
from safetensors.numpy import save_file
from safetensors import safe_open


def OpenFile(file):
    path = os.path.join(os.path.dirname(__file__), file)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def Tokenize(text, method="words", by=None):
    if method == "words":
        return text.split(by)


def Probs(words):
    probs = {}

    for word in words:
        probs[word] = probs.get(word, 0) + 1

    return probs

def p(probs, model, method="cpu", dtype="float32"):
    """
    P operation to calculate probabilities.\n\n**probs** = Dict\n\n**model** = model\n\n**method** = cpu, cuda\n\n**dtype** = float32, float64, etc
    """

    if method == "cpu":
        p = numpy.array(
            list(probs.values()),
            dtype=dtype
        )

        p = p ** (model.tdim / 32)
        p /= p.sum()
        return p
    if method == "cuda":
        p = cupy.array(
                    list(probs.values()),
                    dtype=dtype
                )
        
        p = p ** (model.tdim / 32)
        p /= p.sum()
        return p.get()

class SaveTensor:
    
    @staticmethod
    def Bigram(model, method, dtype="float32", filename="model.safetensors"):
        """
        A simple function to save a Bigram model as a safetensors file. 
        
        **Method** = cuda, cpu 
        
        **model** = your model
         
        **filename** = filename
        """
        
        if method == "cuda":
            mdiv = cupy.array([model.mdiv, model.dim, model.batch], dtype=dtype)
            tdiv = cupy.array([model.ddiv, model.mdiv], dtype=dtype)
            tdim = cupy.array([model.dim, model.ddiv], dtype=dtype)
            
            tensors = {"mdiv": cupy.asnumpy(mdiv), "tdiv": cupy.asnumpy(tdiv), "tdim": cupy.asnumpy(tdim)}
            save_file(tensors, filename)
        if method == "cpu":
            mdiv = numpy.array([model.mdiv, model.dim, model.batch], dtype=dtype)
            tdiv = numpy.array([model.ddiv, model.mdiv], dtype=dtype)
            tdim = numpy.array([model.dim, model.ddiv], dtype=dtype)
            
            tensors = {"mdiv": numpy.asarray(mdiv), "tdiv": numpy.asarray(tdiv), "tdim": numpy.asarray(tdim)}
            save_file(tensors, filename)