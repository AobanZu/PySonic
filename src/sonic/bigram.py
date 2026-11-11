class Model:
    def __init__(self, dim, batch, ddiv=32, mdiv=32):
        self.dim = dim
        self.batch = batch
        self.ddiv = ddiv
        self.mdiv = mdiv
        
    @staticmethod
    def Parameters(tdim, words):
        """
        Returns bigram Parameters
        """
        
        print(f"True Dimensions: {tdim}")
        print(f"Vocab: {len(words)}")

def InitModel(model):
    """
    # sonic.bigram.InitModel
    **WARNING: Only use this model for GENERATION**
    
    Creates a temporary model that can generate text.
    Put in your model from sonic.bigram.Model as an argument.
    """
    
    if model.batch == model.dim or model.batch == model.mdiv:
        raise ValueError("Model size invalid")

    model.mdiv = int(model.mdiv * model.dim / model.batch)

    model.tdiv = model.ddiv * model.mdiv
    model.tdim = round(model.dim / model.ddiv)
    
    return model
