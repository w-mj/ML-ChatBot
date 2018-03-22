import re


class Separator(object):
    def __init__(self):
        self.dictionary = [{}]
        self.max_word_length = 0
        with open('dict.txt', encoding='UTF-8') as f:
            for line in f:
                line = line.split()
                self.max_word_length = max(self.max_word_length, len(line[0]))
                dict_cursor = self.dictionary
                for c in line[0]:
                    if c not in dict_cursor[0]:
                        dict_cursor[0][c] = [{}]
                    dict_cursor = dict_cursor[0][c]
                dict_cursor.append(int(line[1]))
                dict_cursor.append(line[2])
        print('load dictionary finished')

    def in_dict(self, word):
        #  认为所有的空字符串都不在字典里，所有的单字都在字典里
        if len(word) == 0:
            return False
        if len(word) == 1:
            return True

        dict_cursor = self.dictionary
        for c in word:
            if c in dict_cursor[0]:
                dict_cursor = dict_cursor[0][c]
            else:
                return False
        return True

    def separate_line(self, line):
        j = len(line)
        if j > self.max_word_length:
            i = j - self.max_word_length
        else:
            i = 0
        # 逆向最大匹配
        reverse_matching = []
        reverse_single = 0
        while j > 0:
            if self.in_dict(line[i: j]):
                reverse_matching.insert(0, line[i: j])
                if j - i == 1:
                    reverse_single += 1
                j = i
                if j > self.max_word_length:
                    i = j - self.max_word_length
                else:
                    i = 0
            else:
                if i < j:
                    i += 1
        # 正向最大匹配
        matching = []
        single = 0
        i = 0
        if i + self.max_word_length < len(line):
            j = i + self.max_word_length
        else:
            j = len(line)
        while i < len(line):
            if self.in_dict(line[i: j]):
                matching.append(line[i: j])
                if j - i == 1:
                    single += 1
                i = j
                if i + self.max_word_length < len(line):
                    j = i + self.max_word_length
                else:
                    j = len(line)
            else:
                if i < j:
                    j -= 1
        if single < reverse_single:
            return matching
        else:
            return reverse_matching

    def separate(self, sentence):
        pattern = re.compile(r'([A-za-z0-9\s,.\';!，。？！、”“‘’%(){\}\[\]]+)')
        sp = pattern.split(sentence)
        result = []
        for e in sp:
            if len(e) == 0:
                continue
            if pattern.fullmatch(e) is None:
                result.extend(self.separate_line(e))
            else:
                result.extend(e)
        return result


if __name__ == '__main__':
    a = Separator()
    for c in a.separate("双向最大匹配法是将正向最大匹配法得到的分词结果和逆向最大匹配法的到的结果进行比较，从而决定正确的分词方法。"
                        "研究表明，中文中90.0％左右的句子，正向最大匹配法和逆向最大匹配法完全重合且正确，只有大概9.0"
                        "％的句子两种切分方法得到的结果不一样，但其中必有一个是正确的（歧义检测成功），只有不到1.0"
                        "％的句子，或者正向最大匹配法和逆向最大匹配法的切分虽重合却是错的，或者正向最大匹配法和逆向最大匹配法切分不同但两个都不对（歧义检测失败）。这正是双向最大匹配法在实用中文信息处理系统中得以广泛使用的原因所在。"):
        print(c, end=' ')
