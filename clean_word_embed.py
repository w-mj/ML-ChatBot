import io
import numpy as np


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ', maxsplit=1)
        # data[tokens[0]] = map(float, tokens[1:])
        data[tokens[0]] = line[len(tokens[0]) + 1: ]
    return data, n, d


def load_dicts(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = set()
    for line in fin:
        data.add(line.split()[0])
    return data


if __name__ == '__main__':
    vec, n, d = load_vectors('wiki.zh.vec')
    print('open vec')
    dic = load_dicts('common_words.txt')
    print('open dict')
    veck = set(vec.keys())
    res = dic & veck  # 交集
    randvec = np.random.randn(4, d)
    np.set_printoptions(precision=3, suppress=True, linewidth=10000)
    print('write')
    with open('word_embed_clean.vec', 'w', encoding='utf-8') as f:
        f.write('{} {}\n'.format(len(res) + 4, d))
        f.write('UNK {}\n'.format(str(randvec[0])[1: -2]))
        f.write('GO {}\n'.format(str(randvec[1])[1: -2]))
        f.write('PAD {}\n'.format(str(randvec[2])[1: -2]))
        f.write('EOS {}\n'.format(str(randvec[3])[1: -2]))
        for k in res:
            f.write(k + ' ' + vec[k])
