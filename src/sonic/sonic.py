import os
import re
import cupy
import random
import numpy
from safetensors.numpy import save_file, load_file
from safetensors import safe_open
from types import SimpleNamespace


def OpenFile(file):
    """
    # Open File
    
    Read a text file and return its contents as a string.
    
    ### Parameters
    
    - **file** (str): A filename located in the root directory or a file path leading to it.
    
    ### Output
    Returns a string in UTF-8 encoding.
    """

    
    path = os.path.join(os.path.dirname(__file__), file)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def softmax(x, method="cpu"):
    """Compute softmax values for each set of scores in x using CPU (NumPy) or CUDA (CuPy)."""
    if method == "cuda":
        xp = cupy
        x = xp.asarray(x)
    elif method == "cpu":
        xp = numpy
        if cupy is not None and isinstance(x, cupy.ndarray):
            x = cupy.asnumpy(x)
    else:
        raise ValueError(f"Invalid method '{method}'. Expected 'cpu' or 'cuda'.")

    e_x = xp.exp(x - xp.max(x, axis=0, keepdims=True))
    result = e_x / xp.sum(e_x, axis=0, keepdims=True)

    if method == "cuda":
        return cupy.asnumpy(result)

    return result




def Tokenize(text, method="words", by=None):
    """
    # Tokenize
    Split text into tokens, currently supporting two methods "num" and "words".
    
    ### Parameters
    - **text**(`str`): The plaintext to convert to tokens.
    - **method**(`str`): Methods to use, supports "num" and "words"
    - **by**(`str`): The seperator to split the text by. 
    
    ### Returns
    
    - **Words**: Returns a list of strings.
    - **Num**: Returns vocab(x), and the text turned into tokens(z)
    """

        
    if method == "words":
        return text.split(by)
    
    if method == "num":
        split = text.split(by)
        vc = {}
        z = []
        for word in split:
            if word not in vc:
                vc[word] = len(vc)   # next unused id
            z.append(vc[word])
        return vc, z
            
            
            


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
            w1 = cupy.asarray(model.w1, dtype=dtype)
            bias = cupy.array([model.bias], dtype=dtype)
            
            tensors = {"mdiv": cupy.asnumpy(mdiv), "tdiv": cupy.asnumpy(tdiv), "tdim": cupy.asnumpy(tdim), "vc": cupy.asnumpy(vocab), "w1": cupy.asnumpy(w1), "bias": cupy.asnumpy(bias)}
            save_file(tensors, filename)
        if method == "cpu":
            mdiv = numpy.array([model.mdiv, model.dim, model.batch], dtype=dtype)
            tdiv = numpy.array([model.ddiv, model.mdiv], dtype=dtype)
            tdim = numpy.array([model.dim, model.ddiv], dtype=dtype)
            vocab = numpy.array([model.vocab], dtype=dtype)
            w1 = model.w1
            bias = numpy.array([model.bias], dtype=dtype)
            
            tensors = {"mdiv": numpy.asarray(mdiv), "tdiv": numpy.asarray(tdiv), "tdim": numpy.asarray(tdim), "vc": numpy.asarray(vocab), "w1": numpy.asarray(w1), "bias": numpy.asarray(bias)}
            save_file(tensors, filename)
class LoadTensor:
    
    @staticmethod
    def Bigram(filename="model.safetensors", method="cpu", dtype="float32"):
        import bigram
        tensors = load_file(filename)
        
        mdiv_val, dim_val, batch_val = tensors["mdiv"].reshape(-1).tolist()
        ddiv_val, _ = tensors["tdiv"].reshape(-1).tolist()
        vocab_val, = tensors["vc"].reshape(-1).tolist()
        vocab = int(vocab_val)
        
        print("Loaded params -> dim:", dim_val, "batch:", batch_val, "mdiv:", mdiv_val, "ddiv:", ddiv_val, "vocab:", vocab)
        
        model = bigram.Model(
            int(dim_val),
            int(batch_val),
            words=[0] * vocab,
            ddiv=int(ddiv_val),
            mdiv=int(mdiv_val)
        )
        
        if "bias" in tensors:
            bias_val, = tensors["bias"].reshape(-1).tolist()
            model.bias = float(bias_val)
            
        if "w1" in tensors:
            if method == "cuda":
                model.w1 = cupy.asarray(tensors["w1"], dtype=dtype)
            elif method == "cpu":
                model.w1 = numpy.asarray(tensors["w1"], dtype=dtype)
                
        return model
