from gramformer import Gramformer
import torch


import difflib

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

    def find_deltas(self, sentence, corrected):
        """
        Takes in the original sentence and the corrected sentence

        Returns an array of dictionaries indicating the type of change, starting index and ending index (not-inclusive) of non-punctuation changes in the corrected sentence from the original sentence.

        Example:
        `find_delta("this is the original sentence", "This is the corrected sentence.")`
        returns `[{'type': 'ADJ', 'start': 12, 'end': 21}]`       
        """
        deltas = []
        edits = self._gf.get_edits(sentence, corrected)

        # Only show edits that are not about punctuation
        edits = [edit for edit in edits if (edit[0] != "NOUN" and edit[0] != "ORTH" and edit[0] != "SPELL" and edit[0] != "OTHER")]
        for edit in edits:
          pos = self._character_position_from_word_position(corrected.split(" "), edit[5], edit[6])
          deltas.append({
            "type": edit[0],
            "start": pos[0],
            "end": pos[1]
          })

        return deltas

    @staticmethod
    def _character_position_from_word_position(split_sentence, start, end):
      """
      Takes in a list of words (assumed to only be separated by one space character), a starting word position, and an ending word position (not-inclusive).

      Returns the start and end positions of the segment of word(s) in the original sentence.

      Example:
      `character_position_from_word_position(["This", "is", "a", "sentence"], 2, 4)`
      returns `(8, 18)` 
      """

      # start off with the number of spaces
      char_start = start 
      char_end = end-start

      for i in range(start):
        char_start += len(split_sentence[i])

      for i in range(end-start):
        char_end += len(split_sentence[start+i])
      char_end += char_start - 1

      return (char_start, char_end)
      



def testingMethod():
    """
    Method to demonstrate usage of grammar_corrector
    """
    set_seed(5279)
    gc = Grammar_Corrector()
    
    sentence = "Thsi sentence am incorrect. it's has bads spelling missing commas no period and other issues"
    corrected = gc.correct_sentence(sentence)
    print(sentence)
    print(corrected)

    deltas = gc.find_deltas(sentence, corrected)
    # print(deltas) # Can be printed for debugging
    correction_underline = [" "]*len(corrected)
    for delta in deltas:
      for i in range(delta["end"]-delta["start"]):
          correction_underline[delta["start"]+i] = "^"

    correction_underline = "".join(correction_underline)
    print(correction_underline)
