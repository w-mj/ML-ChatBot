import io
import numpy as np


# 判断一个unicode是否是汉字
def is_chinese(uchar):
    if '\u4e00' <= uchar <= '\u9fff':
        return True
    else:
        return False


# 判断一个unicode是否是数字
def is_number(uchar):
    if '\u0030' <= uchar <= '\u0039':
        return True
    else:
        return False


# 判断一个unicode是否是英文字母
def is_alphabet(uchar):
    if ('\u0041' <= uchar <= '\u005a') or ('\u0061' <= uchar <= '\u007a'):
        return True
    else:
        return False


# 判断是否非汉字，数字和英文字符
def is_other(uchar):
    if not (is_chinese(uchar) or is_number(uchar)):
        return True
    else:
        return False


def is_useful(uchar):
    return not is_other(uchar)


def is_str_useful(ustr):
    for x in ustr:
        if is_other(x):
            return False
    return True


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


def intersection(vec):
    dic = load_dicts('common_words.txt')
    print('open dict')
    veck = set(vec.keys())
    res = dic & veck  # 交集
    return res


def judge(vec):
    veck = set(vec.keys())
    res = set(filter(is_str_useful, veck))
    return res


def write_file(res, vec, n, d):
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

def single_word(vec, n, d):
    with open('usually_word', encoding='utf-8') as f:
        characters = f.read()

    charac = set(characters)
    res = set()
    for word in vec.keys():
        if set(word).issubset(charac):
            res.add(word)
    return res

def main():
    vec, n, d = load_vectors('wiki.zh.vec')
    res = single_word(vec, n, d)
    write_file(res, vec, n, d)



if __name__ == '__main__':
    main()