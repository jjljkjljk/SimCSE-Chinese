from dataset import MyDataset
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import argparse
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from scipy.stats import spearmanr


def load_test_data(tokenizer, args):
    """
    加载测试集
    """
    logger.info('loading data')
    feature_list = []
    with open(args.test_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            _, sent1, sent2, score = line.strip().split('||')
            score = float(score)
            sent1 = tokenizer(sent1, max_length=args.max_len, truncation=True, padding='max_length',
                              return_tensors='pt')
            sent2 = tokenizer(sent2, max_length=args.max_len, truncation=True, padding='max_length',
                              return_tensors='pt')
            feature_list.append((sent1, sent2, score))
    logger.info("len of test data:{}".format(len(feature_list)))
    return feature_list


def evaluate(model, dataloader, device):
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in tqdm(dataloader):
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(device)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids, output_hidden_states=True, return_dict=True)
            source_pred = source_pred.last_hidden_state[:, 0]
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(device)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids, output_hidden_states=True, return_dict=True)
            target_pred = target_pred.last_hidden_state[:, 0]
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="gpu or cpu")
    parser.add_argument("--batch_size_eval", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=100, help="max length of input")
    parser.add_argument("--test_file", type=str, default="data/STS-B/test.txt")
    parser.add_argument("--pretrain_model_path", type=str, default="pretrain_model/bert-base-chinese")
    parser.add_argument("--do_eval", action='store_true', default=True)
    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    bert = BertModel.from_pretrained(args.pretrain_model_path).to(args.device)
    bert.eval()
    test_data = load_test_data(tokenizer, args)
    # test_data = test_data[:8]
    test_dataset = MyDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=True,
                                 num_workers=args.num_workers)

    corrcoef = evaluate(bert, test_dataloader, args.device)
    logger.info('corrcoef of bert:{}'.format(corrcoef))