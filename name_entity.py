from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from os import walk, path
from collections import Counter
from sys import platform
import string
from nltk.stem.snowball import SnowballStemmer

sentence = 'Mark and John are working at Google'

ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))

iob_tagged = tree2conlltags(ne_tree)

ne_tree2 = conlltags2tree(iob_tagged)

if platform == 'linux' or platform == 'linux2':
    corpus_root = '/media/markxueyuan/Data/gmb-2.2.0'
elif platform == 'darwin':
    pass
elif platform == 'win32':
    corpus_root = 'D:/Corpus/gmb-2.2.0'

name_tags = Counter()

for root, dirs, files in walk(corpus_root):
    for filename in files:
        if filename.endswith('.tags'):
            with open(path.join(root, filename), 'rb') as file_handle:
                file_content = file_handle.read().decode('utf-8').strip()
                sentences = file_content.split('\n\n')
                for sentence in sentences:
                    tokens = [seq for seq in sentence.split('\n') if seq]
                    standard_tokens = []
                    for indx, token in enumerate(tokens):
                        annotations = token.split('\t')
                        word, tag, ner = annotations[0], annotations[1], annotations[3]

                        if ner != '0':
                            ner = ner.split('-')[0]
                        name_tags[ner] += 1

print(name_tags)

def features(tokens, index, history):
    """
    :param tokens: a POS-tagged sentence [(w1, t1), ...]
    :param index: the index of the token we want to extract features for
    :param history: the previous predicted IOB tags
    :return:
    """

    stemmer = SnowballStemmer('english')

    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)

    index += 2

    word, pos = tokens[index]
