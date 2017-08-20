from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from os import walk, path
from collections import Counter
from sys import platform
import string
from nltk.stem.snowball import SnowballStemmer
import pickle
from collections import Iterable
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI
import pdb
import csv
from time import time

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

# name_tags = Counter()
#
# for root, dirs, files in walk(corpus_root):
#     for filename in files:
#         if filename.endswith('.tags'):
#             with open(path.join(root, filename), 'rb') as file_handle:
#                 file_content = file_handle.read().decode('utf-8').strip()
#                 sentences = file_content.split('\n\n')
#                 for sentence in sentences:
#                     tokens = [seq for seq in sentence.split('\n') if seq]
#                     standard_tokens = []
#                     for indx, token in enumerate(tokens):
#                         annotations = token.split('\t')
#                         word, tag, ner = annotations[0], annotations[1], annotations[3]
#
#                         if ner != '0':
#                             ner = ner.split('-')[0]
#                         name_tags[ner] += 1
#
# print(name_tags)

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

    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]

    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]

    previob = history[index - 1]

    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])

    allcaps = word == word.upper()
    capitalized = word[0] in string.ascii_uppercase

    prevallcaps = prevword == prevword.upper()
    prevcapitalized = prevword[0] in string.ascii_uppercase

    nextallcaps = nextword == nextword.upper()
    nextcapitalized = nextword[0] in string.ascii_uppercase

    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all-ascii': allascii,

        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,
        'next-next-word': nextnextword,
        'next-next-pos': nextnextpos,

        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,

        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,

        'prev-iob': previob,

        'contains-dash': contains_dash,
        'contains-dot': contains_dot,

        'all-caps': allcaps,
        'capitalized': capitalized,

        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,

        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized

    }


def to_conll_iob(annotated_sentence):
    """

    :param annotated_sentence: list of triplets [(w1, t1, iob1), ...]
    Transform a pseudo-IOB notation: 0, person, person, 0, 0, , location, 0
    to proper IOB notation: 0, B-PERSON, I-PERSON, 0, 0, B-LOCATION, 0

    :return:
    """

    proper_iob_tokens = []

    for idx, annotated_token in enumerate(annotated_sentence):

        word, tag, ner = annotated_token

        if ner != 'O':
            if idx == 0:
                ner = 'B-' + ner
            elif annotated_sentence[idx-1][2] == ner:
                ner = 'I-' + ner
            else:
                ner = 'B-' + ner
        proper_iob_tokens.append((word, tag, ner))

    return proper_iob_tokens

def read_gmb(corpus_root):
    for root, dirs, files in walk(corpus_root):
        for filename in files:
            if filename.endswith('.tags'):
                with open(path.join(root, filename), 'rb') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]
                        standard_form_tokens = []

                        for idx, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[0], annotations[1], annotations[3]

                            if ner != '0':
                                ner = ner.split('-')[0]

                            if tag in ('LQU', 'RQU'):
                                tag = "``"

                            standard_form_tokens.append((word, tag, ner))



                        conll_tokens = to_conll_iob(standard_form_tokens)

                        yield [((w, t), iob) for w, t, iob in conll_tokens]



class NamedEntityChuncker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)
        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=features,
            **kwargs)

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        return conlltags2tree(iob_triplets)

    def parse2(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        return iob_triplets


reader = read_gmb(corpus_root)


data = list(reader)

training_samples = data[:int(len(data) * 0.9)]
test_samples = data[int(len(data) * 0.9):]

print("#training samples = %s" % len(training_samples))
print("#test samples = %s" % len(test_samples))

chunker = NamedEntityChuncker(training_samples)


test = pos_tag(word_tokenize("Apple Corporation is a good company."))

chunker.parse(test)

# tokens = []
# history = []
# cnt = 0
# for item in reader.next():
#     token, iob = item
#     tokens.append(token)
#     history.append(iob)
#     cnt += 1
#
# ft = [features(tokens, idx, history) for idx in range(cnt)]

datapath = "D:/data/Lexi/output.csv"

new_results =[]
start = time()
with open(datapath, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for indx, row in enumerate(reader):
        if indx % 50 == 0:
            print(indx)
        rslt = pos_tag(word_tokenize(row[1]))
        rslt = chunker.parse2(rslt)
        v = []
        for t in rslt:
            v.append(" ".join(t))
        rslt = "---".join(v)

        new_results.append([row[0], row[1], row[2], rslt])

end = time()

print(end - start)

with open("D:/data/Lexi/output_xue.csv", 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for row in new_results:
        writer.writerow(row)
