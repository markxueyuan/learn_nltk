from nltk.corpus.reader import WordListCorpusReader
# WordListCorpusReader inherits from CorpusReader
# CorpusReader is a common base class for all corpus readers

# WordListCorpusReader reads the files and tokenizes each line
# to produce a list of words.

# CorpusReader: fileids()
# WordListCorpusReader(CorpusReader): words()
#     when call the words() function, it calls nltk.tokenize.line_tokenize()
#     on the raw file data

from nltk.corpus.reader import TaggedCorpusReader



filename ="wordlist.txt"
reader = WordListCorpusReader('.', [filename])
print(reader.words())
print(reader.fileids())
reader.raw()

# stopwords is an instance of WordListCorpusReader

from nltk.corpus import stopwords
stopwords.words()
stopwords.words('English')
