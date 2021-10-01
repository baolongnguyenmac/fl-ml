import json
import re
import numpy as np

VOCAB_DIR = 'embs.json'

def split_line(line):
        '''split given line/phrase into list of words
        Args:
            line: string representing phrase to be split
        
        Return:
            list of strings, with each string representing a word
        '''
        return re.findall(r"[\w']+|[.,!?;]", line)

def get_word_emb_arr(path):
    with open(path, 'r') as inf:
        embs = json.load(inf)
    vocab = embs['vocab']
    word_emb_arr = np.array(embs['emba']).tolist()
    indd = {}
    for i in range(len(vocab)):
        indd[vocab[i]] = i
    vocab = {w: i for i, w in enumerate(embs['vocab'])}
    return word_emb_arr, indd, vocab

def line_to_indices(line, word2id, word_emb_arr, max_words=25):
    '''converts given phrase into list of word indices
        
    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer 
    representing unknown index to returned list until the list's length is 
    max_words
    Args:
        line: string representing phrase/sequence of words
        word2id: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list
    Return:
        indl: list of word indices, one index for each word in phrase
    '''
    unk_id = len(word2id)
    line_list = split_line(line) # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id]*(max_words-len(indl))
    emba = []
    for i in range(0, len(indl)):
        emba.append(word_emb_arr[indl[i]])
    return emba

def embedding(path_in, path_out):
    '''
    path_in: raw json data
    path_out: save embedding json data
    '''
    word_emb_arr, indd, _ = get_word_emb_arr(VOCAB_DIR)
    
    with open(path_in, 'r') as openfile:
        data = json.load(openfile)

    for user in data['user_data']:
        for idx, sentence in enumerate(data['user_data'][user]['x']):
            data['user_data'][user]['x'][idx] = line_to_indices(sentence[4], indd, word_emb_arr)

    with open(path_out, "w") as outfile:
        json.dump(data, outfile)
