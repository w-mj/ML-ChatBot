import re
import jieba
import threading


class Vocabulary(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        self._word2index = {}
        self._index2word = {}
        with open('words.txt', encoding='utf-8') as f:
            f.readline()
            for (i, line) in enumerate(f):
                self._word2index[line[:-1]] = i
                self._index2word[i] = line[:-1]
        self.word_num = len(self._word2index)

    def __new__(cls, *args, **kwargs):
        if not hasattr(Vocabulary, '_instance'):
            with Vocabulary._instance_lock:
                if not hasattr(Vocabulary, '_instance'):
                    Vocabulary._instance = object.__new__(cls)
        return Vocabulary._instance

    @property
    def EOS(self):
        return self._word2index['<EOS>']

    @property
    def SOS(self):
        return self._word2index['<SOS>']

    @property
    def UNK(self):
        return self._word2index['<UNK>']

    @property
    def PAD(self):
        return self._word2index['<PAD>']

    def word2index(self, word):
        return self._word2index.get(word, self.UNK)

    def index2word(self, index):
        return self._index2word.get(index, '<OUT>')

    def words2indexs(self, words):
        return [self.word2index(x) for x in words]

    def indexs2words(self, indexs):
        return [self.index2word(x) for x in indexs]

    @staticmethod
    def clean_text(text, max_words=None):
        text = re.sub(r"[()\"#/@;；:：<《>》{}`‘’'+=~|$&*%\[\]_]", " ", text)
        text = re.sub(r"[.]+", " . ", text)
        text = re.sub(r"[!]+", " ! ", text)
        text = re.sub(r"[?]+", " ? ", text)
        text = re.sub(r"[,-]+", " ", text)
        text = re.sub(r"[。]+", " . ", text)
        text = re.sub(r"[！]+", " ! ", text)
        text = re.sub(r"[？]+", " ? ", text)
        text = re.sub(r"[，-]+", " ", text)
        text = re.sub(r"[\t]+", " ", text)
        text = re.sub(r" +", " ", text)
        text = text.strip()

        # Truncate words beyond the limit, if provided.
        if max_words is not None:
            text_parts = text.split()
            if len(text_parts) > max_words:
                text = " ".join(text_parts[:max_words])

        if text == '':
            text = "?"

        return text

    def prepare_text(self, text, cut=False, sos=False):
        text = self.clean_text(text)
        if cut:
            text = jieba.cut(text)
        else:
            text = text.split()
        text = self.words2indexs(text)
        text.append(self.EOS)
        if sos:
            text.insert(0, self.SOS)
        return text

