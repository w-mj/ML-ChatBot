import io


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
    print('write')
    special_embedding = '0 ' * d
    with open('word_embed_clean.vec', 'w', encoding='utf-8') as f:
        f.write('{} {}\n'.format(len(res) + 4, d))
        f.write('UNK {}\n'.format(special_embedding))
        f.write('GO {}\n'.format(special_embedding))
        f.write('PAD {}\n'.format(special_embedding))
        f.write('EOS {}\n'.format(special_embedding))
        for k in res:
            f.write(k + ' ' + vec[k])
