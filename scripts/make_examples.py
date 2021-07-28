from argparse import ArgumentParser
from itertools import combinations, permutations
from operator import itemgetter
import random

import torch
import numpy as np
from tqdm import tqdm

from sentence_transformers import InputExample

from make_model import make_model


def load_sentence_and_label(sent_paths, cls_paths):
    sentence_list = []
    for sent_path in sent_paths:
        with open(sent_path) as sent_file:
            for sent in sent_file:
                sentence_list.append(sent.strip())
    label_list = []
    for cls_path in cls_paths:
        with open(cls_path) as cls_file:
            for label in cls_file:
                label = label.strip()
                if label.isdecimal():
                    label_list.append(int(label))
                else:
                    label_list.append(str(label))
    
    label_vary = sorted(set(label_list))
    label_dict = { label:idx for idx, label in enumerate(label_vary) }
    labels = [ label_dict[label] for label in label_list ]
    
    return sentence_list, labels


def make_single_examples(sentence_list, label_list):
    """
    Making single examples : example([text], label) {sentence and class}
    """
    print("[INFO] Making single data ... from length {}".format(len(sentence_list)))
    examples = []
    for sent, label in zip(sentence_list, label_list):
        examples.append(InputExample(texts=[sent], label=label))
    print("[INFO] Finish making single data {}".format(len(examples)))
    return examples


def make_double_examples(sentence_list, label_list, embed_list=None, margin=5.0):
    """
    Making double examples : example([text, text], label) {double sentences and similarity}
    """
    print("[INFO] Making double data ... from length {}".format(len(sentence_list)))
    examples = []
    negative_data_list = []
    
    # random
    if embed_list is None:
        all_data_list = [ [sent, label] for sent, label in zip(sentence_list, label_list) ]
        for data_a, data_b in tqdm(list(combinations(all_data_list, 2)), ascii=True, dynamic_ncols=True):
            if data_a[1] == data_b[1]:
                examples.append(InputExample(texts=[data_a[0], data_b[0]], label=1.0))
            else:
                negative_data_list.append([data_a[0], data_b[0], 0])
        negative_data_list = random.sample(negative_data_list, len(examples))
        
    # hard
    else:
        all_data_list = [ [sent, label, embed] for sent, label, embed in zip(sentence_list, label_list, embed_list) ]
        for data_a, data_b in tqdm(list(combinations(all_data_list, 2)), ascii=True, dynamic_ncols=True):
            if data_a[1] == data_b[1]:
                # positive
                examples.append(InputExample(texts=[data_a[0], data_b[0]], label=1.0))
            else:
                # negative
                negative_data_list.append([data_a[0], data_b[0], np.linalg.norm(data_a[2]-data_b[2])])
        negative_data_list = sorted(negative_data_list, key=itemgetter(2))[:len(examples)]
    
    for sent_a, sent_b, similarity in negative_data_list:
        examples.append(InputExample(texts=[sent_a, sent_b], label=0.0))
    print("[INFO] Finish making double data {}".format(len(examples)))
    
    return examples


def make_triple_examples(sentence_list, label_list, embed_list=None, margin=5.0):
    """
    Making triple examples : example([text, text, text]) {triple sentences}}
    """
    def random_from(data_a, neg_lst):
        return random.choice(neg_lst)
        
    def hard_from(data_a, neg_lst):
        anc_embed = torch.tensor(data_a[1]).unsqueeze(0)
        neg_embed = torch.tensor([ data_n[1] for data_n in neg_lst ])
        distance = torch.norm(anc_embed - neg_embed, dim=-1)
        return neg_lst[distance.argmin()]
    
    print("[INFO] Making triple data ... from length {}".format(len(sentence_list)))
    examples = []
    label_vary = len(set(label_list))
    label_data_list = [ [] for i in range(label_vary) ]
    if embed_list is None:
        sample_from = random_from
        for sent, label in zip(sentence_list, label_list):
            label_data_list[label].append([sent])
    else:
        sample_from = hard_from
        for sent, label, embed in zip(sentence_list, label_list, embed_list):
            label_data_list[label].append([sent, embed])
    
    for pos_idx in tqdm(range(label_vary), ascii=True, dynamic_ncols=True):
        tmp_pos_list = label_data_list[pos_idx]
        tmp_neg_list = []
        for neg_idx in range(label_vary):
            if neg_idx != pos_idx:
                tmp_neg_list += label_data_list[neg_idx]
        for data_a, data_p in permutations(tmp_pos_list, 2):
            data_n = sample_from(data_a, tmp_neg_list)
            examples.append(InputExample(texts=[data_a[0], data_p[0], data_n[0]]))
    print("[INFO] Finish making triple data {}".format(len(examples)))
    
    return examples


def main():
    
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--pooling_strategy", type=str, default="mean", choices=["mean", "cls", "max"])
    parser.add_argument("--load_sent", type=str, nargs="+", default=[])
    parser.add_argument("--load_cls",  type=str, nargs="+", default=[])
    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data_type", type=str, default="double", choices=["single", "double", "triple"])
    parser.add_argument("--sampling_strategy", type=str, default="random", choices=["random", "hard", "semi-hard"])
    parser.add_argument("--triplet_margin", type=float, default=5.0)
    opt = parser.parse_args()
    
    # set seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    
    # load data
    sentence_list, label_list = load_sentence_and_label(opt.load_sent, opt.load_cls)
    
    # set data type
    if opt.data_type == "single":
        make_data = make_single_examples
    elif opt.data_type == "double":
        make_data = make_double_examples
    elif opt.data_type == "triple":
        make_data = make_triple_examples
    
    # make data
    if opt.sampling_strategy == "random":
        examples = make_data(sentence_list, label_list)
    elif opt.sampling_strategy == "hard":
        # compute sentence embeddings
        model = make_model(opt.model_name_or_path, opt.pooling_strategy)
        embed_list = model.encode(sentence_list)
        examples = make_data(sentence_list, label_list, embed_list=embed_list, margin=opt.triplet_margin)
    
    # save data
    torch.save(examples, opt.save_file)
    
    
if __name__ == "__main__":
    
    main()

