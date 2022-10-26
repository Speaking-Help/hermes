from gramformer import Gramformer
import torch
import spacy

def set_seed(seed):
    """
    Sets the seed for PyTorch's RNG

    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Grammar_Corrector:
    def __init__(self): 
        """
        Loads the Gramformer model
        """
        self._gf = Gramformer(models=1, use_gpu=False)

    def correct_sentence(self, sentence):
        """
        Passes the sentence into Gramformer's correct function, with one candidate

        Returns the corrected sentence
        """
        corrected_sentences = self._gf.correct(sentence, max_candidates=1)
        return corrected_sentences.pop()    #only one value is present, pop it from set


def main():
    """
    Method to demonstrate usage of grammar_corrector
    """
    set_seed(5279)
    gc = Grammar_Corrector()
    sentence = gc.correct_sentence("Thsi sentence is incorrect. it has bad spelling missing commas no period and other issues")
    print(sentence)

if __name__ == "__main__":
    main()
