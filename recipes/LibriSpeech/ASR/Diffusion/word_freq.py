import os

import torch
import numpy as np
import math
from tqdm import tqdm
import logging
import speechbrain as sb
from functools import partial
from datasets import load_dataset

"""
Authors
 * Pooneh Mousavi, 2023
"""

logger = logging.getLogger(__name__)


def prepare_word_frequencies(data_file,data_folder,save_folder, tokenizer):
    """
    This converts lines of text into a dictionary of word ferquencies-

    Arguments
    ---------
    split : str
        name to the file containing the librispeech text transcription.
    data_folder: str
        path to the file containing the librispeech text transcription.
    save_folder: str
        path to the file to save the word frequencies dictionary
    tokenizer: Bert or Roberta Tokenizer
        Tokenizer to encode the texts

    Returns
    -------
    save the dictionary contains frequency for each words in the data in {save_folder}/word_freq.pt

    """


    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_file= os.path.join(save_folder, "word_freq.pt")

    if os.path.isfile(save_file):
        logger.info("Skipping preparation, completed in previous run.")
        return
    
    if not (os.path.isfile(data_file)):
        logger.error(f"{data_file} doesn not exist!!!")
        return

    train_data = TextLoader(tokenizer=tokenizer).load(data_file=data_file)

    word_freq = torch.zeros((tokenizer.vocab_size,), dtype=torch.int64)
    i=0
    for data in tqdm(train_data):
        i+=1
        for iid in data['input_ids']:
            word_freq[iid] += 1
        if i >=  100:
            break
    torch.save(word_freq, save_file)



class TextLoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


    def load(self, data_file):
        dataset = load_dataset("csv", data_files=data_file, split='train')
        dataset = dataset.map(partial(self.convert_to_features, tokenizer=self.tokenizer), batched=True, remove_columns='wrd')
        return dataset


    @staticmethod
    def convert_to_features(example_batch, tokenizer):
        input_encodings = tokenizer.batch_encode_plus(example_batch['wrd'], max_length=128, truncation=True, add_special_tokens=False)
        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
        }

        return encodings
    




if __name__ == "__main__":
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    prepare_word_frequencies('train.csv',"./data","./save",tokenizer)

