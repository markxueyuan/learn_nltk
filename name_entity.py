from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from os import walk, path
from collections import Counter

sentence = 'Mark and John are working at Google'

ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))

iob_tagged = tree2conlltags(ne_tree)

ne_tree2 = conlltags2tree(iob_tagged)

corpus_root = '/media/markxueyuan/Data/gmb-2.2.0'

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

