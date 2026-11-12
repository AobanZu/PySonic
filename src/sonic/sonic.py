import os
import re
import cupy
import numpy
from safetensors.numpy import save_file
from safetensors import safe_open


def OpenFile(file):
    """
    # Open File
    
    Read a text file and return its contents as a string.
    
    # Parameters
    
    - **file** (str): A filename located in the root directory or a file path leading to it.
    
    # Output
    Returns a string in UTF-8 encoding.
    """

    
    path = os.path.join(os.path.dirname(__file__), file)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def Tokenize(text, method="words", by=None):
    """
    # Tokenize
    Split text into tokens, using numbers or words.
    
    # Parameters
    - **text**(s): The plaintext to convert to tokens.
    - **method**: Methods to use, supports "num" and "words"
    - **by**
    """
    
    if method == "words":
        return text.split(by)


def Probs(words):
    """
    # Probs
    This function finds probabilities for each word. [!WARNING] >[DEPRECATED] 
    Probabilities are counted from how common they are, this can be used for generation and
    analysis on how the model works. 
    ## Usage: 
    
    Probs(words="The quick brown fox ran over the lazy dog and the dog was angry.), Output:
    
    The : 3
    dog : 2
    over : 1
    ran : 1
    fox : 1
    quick : 1
    lazy : 1
    brown : 1
    was : 1
    angry : 1
    and : 1
    over : 1
    
    """
    
    probs = {}

    for word in words:
        probs[word] = probs.get(word, 0) + 1

    return probs

def p(probs, model, method="cpu", dtype="float32"):
    """
    # P - Function
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
            vocab = cupy.array([model.vocab], dtype=dtype)
            
            tensors = {"mdiv": cupy.asnumpy(mdiv), "tdiv": cupy.asnumpy(tdiv), "tdim": cupy.asnumpy(tdim), "vc": cupy.asnumpy(vocab)}
            save_file(tensors, filename)
        if method == "cpu":
            mdiv = numpy.array([model.mdiv, model.dim, model.batch], dtype=dtype)
            tdiv = numpy.array([model.ddiv, model.mdiv], dtype=dtype)
            tdim = numpy.array([model.dim, model.ddiv], dtype=dtype)
            vocab = numpy.array([model.vocab], dtype=dtype)
            
            tensors = {"mdiv": numpy.asarray(mdiv), "tdiv": numpy.asarray(tdiv), "tdim": numpy.asarray(tdim), "vc": numpy.asarray(vocab)}
            save_file(tensors, filename)