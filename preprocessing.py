import json


def main():
    with open('train_data.txt', encoding='utf-8') as f:
        c = f.read()
        obj = eval(c)
        j = json.dumps(obj, ensure_ascii=False, indent=4)
        with open('train_data.json', 'w', encoding='utf-8') as o:
            o.write(j)


if __name__ == '__main__':
    main()