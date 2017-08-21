from nltk.corpus.reader import WordListCorpusReader
# WordListCorpusReader inherits from CorpusReader
# CorpusReader is a common base class for all corpus readers

# WordListCorpusReader reads the files and tokenizes each line
# to produce a list of words.

# CorpusReader: fileids()
# WordListCorpusReader(CorpusReader): words()
#     when call the words() function, it calls nltk.tokenize.line_tokenize()
#     on the raw file data




filename ="wordlist.txt"
reader = WordListCorpusReader('.', [filename])
print(reader.words())
print(reader.fileids())
reader.raw()

# stopwords is an instance of WordListCorpusReader

from nltk.corpus import stopwords
stopwords.words()
stopwords.words('English')

from nltk.corpus.reader import TaggedCorpusReader

reader = TaggedCorpusReader('.', '.*\.pos')
reader.words()
reader.tagged_words()
reader.tagged_sents()
reader.tagged_paras()



















