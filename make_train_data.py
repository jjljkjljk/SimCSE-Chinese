import json
from collections import defaultdict
import pandas as pd
import random


def get_sup_train_data(file, out_file, label_field):
    """
    从OCNLI数据集中，获取<anchor, entailment, contradiction>三元组
    :param file:
    :return:
    """
    with open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        # {'anchor1':{'entailment':'xxx', 'contradiction':''}, 'anchor2':{'entailment':'xxx', 'contradiction':''}}
        anchor2entail_constra = defaultdict(lambda: defaultdict(str))

        for line in lines:
            line = json.loads(line)
            sentence1 = line['sentence1'].strip()
            sentence2 = line['sentence2'].strip()
            label = line[label_field].strip()
            if label not in ['entailment', 'contradiction']:
                continue
            else:
                anchor2entail_constra[sentence1][label] = sentence2

        result = []
        for anchor in anchor2entail_constra:
            entailment = anchor2entail_constra[anchor]['entailment']
            contradiction = anchor2entail_constra[anchor]['contradiction']
            if entailment.strip() != '' and contradiction.strip() != '':
                result.append({'anchor': anchor, 'entailment': entailment, 'contradiction': contradiction})
        df = pd.DataFrame(result)
        print('nli监督训练集规模：{}'.format(len(df)))
        df.to_csv(out_file, index=False)


def get_unsup_train_data_ocnli(file, out_file):
    with open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()
    lines = random.sample(lines, 20000)  # 随机选20000条数据，作为无监督训练数据

    with open(out_file, 'w', encoding='utf8') as f:
        for line in lines:
            line = json.loads(line)
            sentence1 = line['sentence1'].strip()
            f.write('{}\n'.format(sentence1))


def get_unsup_train_data_stsb(file, out_file):
    with open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()

    count = 0
    with open(out_file, 'w', encoding='utf8') as f:
        for line in lines:
            _, sent1, sent2, score = line.strip().split('||')
            score = int(score)
            if score == 0:
                f.write('{}\n'.format(sent1))
                f.write('{}\n'.format(sent2))
                count += 2
            else:
                f.write('{}\n'.format(sent1))
                count += 1
    print('len of unsup_train_data_stsb:{}'.format(count))


if __name__ == '__main__':
    # 获取cmnli_sup训练集
    file = 'data/cnsd-mnli/cnsd_multil_train.jsonl'
    out_file1 = 'data/cmnli_sup_train_data.csv'
    get_sup_train_data(file, out_file1, 'gold_label')

    # 获取ocnli_sup训练集
    file = 'data/ocnli/train.50k.json'
    out_file1 = 'data/ocnli_sup_train_data.csv'
    get_sup_train_data(file, out_file1, 'label')

    # 获取ocnli_unsup训练集
    file = 'data/ocnli/train.50k.json'
    out_file2 = 'data/ocnli_unsup_train_data.txt'
    get_unsup_train_data_ocnli(file, out_file2)

    # 获取stsb_unsup训练集
    file = 'data/STS-B/train.txt'
    out_file2 = 'data/stsb_unsup_train_data.txt'
    get_unsup_train_data_stsb(file, out_file2)
