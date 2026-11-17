import numpy, cupy
import random

def softmax(x, method="cpu"):
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

class Model:
    """
    # sonic.bigram.Model
    
    (Note, pass words as a dict)

    ## Attributes

    - **dim** (`int`): Base dimension.
    - **batch** (`int`): batch size.
    - **vocab** (`int`): Number of unique tokens.
    - **ddiv** (`int`): Dimension divisor. Default: `32`.
    - **mdiv** (`int`): Median divisor. Default: `32`. Rescaled by
      `InitModel` to `int(mdiv * dim / batch)`. Controls max model output length.

    ## Derived by `InitModel`

    - **tdiv** (`int`): True divisor, `ddiv * mdiv`.
    - **tdim** (`int`): True dimension, `round(dim / ddiv)`.
    """
    def __init__(self, dim, batch, words, ddiv=32, mdiv=32):
        self.dim = dim
        self.batch = batch
        self.vocab = len(words)
        self.ddiv = ddiv
        self.mdiv = mdiv
        
        self.w1 = None
        self.embed = self.vocab * self.dim
        self.bias = float(numpy.random.randint(1, 10) * 0.01) 
               
    @staticmethod
    def Parameters(tdim, vocab):
        """
        Returns bigram Parameters, vocab=dict
        """
        true_dim = tdim * len(vocab)
        print(f"True Dimensions: {true_dim}")
        print(f"Vocab: {len(vocab)}")
        
    def forward(self, probs, model, method="cpu", dtype="float32"):
        """
        # Forward - Function
        forward operation to calculate probabilities.\n\n**probs** = Dict\n\n**model** = temporary model\n\n**method** = cpu, cuda\n\n**dtype** = float32, float64, etc
        """
        
        self.decider = random.randint(0, (len(probs) - 1))

        if method == "cpu":
            p = numpy.array(
                list(probs.values()),
                dtype=dtype
            )
            
            p[self.decider] += self.bias
            
            if self.w1 is not None:
                self.w1 += p
                p = self.w1.copy() 
            else:
                self.w1 = p.copy() 


            p = softmax(p * (model.tdim / 32), method=method)
            return p
        if method == "cuda":
            p = cupy.array(
                        list(probs.values()),
                        dtype=dtype
                    )
            
            p[self.decider] += self.bias
            
            if self.w1 is not None:
                self.w1 += p
                p = self.w1.copy() 
            else:
                self.w1 = p.copy()

                        
            p = softmax(p * (model.tdim / 32), method=method)
            return p
        
    
    

def InitModel(model):
    """
    # sonic.bigram.InitModel
    **WARNING: Only use this model for GENERATION**
    
    Creates a temporary model that can generate text.
    Put in your model from sonic.bigram.Model as an argument.
    """
    if not hasattr(model, "tdim"):
        if model.batch == model.dim or model.batch == model.mdiv:
            raise ValueError("Model size invalid")

        model.mdiv = int(model.mdiv * model.dim / model.batch)

    model.tdiv = model.ddiv * model.mdiv
    model.tdim = round(model.dim / model.ddiv)
    
    return model

