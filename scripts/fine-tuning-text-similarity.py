from argparse import ArgumentParser
import csv
from logging import getLogger, DEBUG, FileHandler, Formatter
import os
import re
import random

import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

import transformers
from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation

from make_model import make_model


def set_logger(file_name, log_level=DEBUG):
    logger = getLogger(__name__)
    fh = FileHandler(file_name)
    fmt = Formatter('[%(levelname)s] %(asctime)s (%(name)s) - %(message)s')
    logger.setLevel(log_level)
    fh.setLevel(log_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class TrainingObserver():
    """
    For early stopping
    """
    def __init__(self, limit_steps, output_path):
        self.limit_steps = limit_steps
        self.min_loss = float('inf')
        self.steps = 0
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.logger = set_logger('{}/valid_loss.log'.format(output_path))

    def check_loss(self, loss, epoch, steps):
        if loss > self.min_loss:
            best = 'keep'
            self.steps += 1
            if self.steps == self.limit_steps:
                print('early stopping')
        else:
            best = 'update'
            self.min_loss = loss
            self.steps = 0
        self.logger.info('epoch: {:2d} steps: {:5d} valid_loss: {:f} best: {:6s} not_improve_cnt: {:d}'.format(epoch, steps, loss, best, self.steps))
        

def main():
    
    parser = ArgumentParser()
    
    # load and save
    parser.add_argument('--model_name_or_path', type=str, default=None)
    parser.add_argument('--train_examples', type=str, default=None)
    parser.add_argument('--valid_examples', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    
    # train
    parser.add_argument('--pooling_strategy', type=str, default='mean', choices=['mean', 'cls', 'max'])
    parser.add_argument('--loss_type', type=str, default='cosine_similarity', choices=['batch_all_triplet',
                                                                                       'batch_hard_soft_margin_triplet',
                                                                                       'batch_hard_triplet',
                                                                                       'batch_semi_hard_triplet',
                                                                                       'cosine_similarity',
                                                                                       'triplet'])
    parser.add_argument('--check_steps', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--evaluation_steps', type=int, default=1000)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--limit_steps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true')
    
    opt = parser.parse_args()
    
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    
    # model (transformer + pooling)
    model = make_model(opt.model_name_or_path, opt.pooling_strategy)
    
    train_examples = torch.load(opt.train_examples)
    valid_examples = torch.load(opt.valid_examples)
    
    # train setting
    if opt.loss_type == 'batch_all_triplet':
        train_loss = losses.BatchAllTripletLoss(model=model)
    elif opt.loss_type == 'batch_hard_soft_margin_triplet':
        train_loss = losses.BatchAllTripletLoss(model=model)
    elif opt.loss_type == 'batch_hard_triplet':
        train_loss = losses.BatchAllTripletLoss(model=model)
    elif opt.loss_type == 'batch_semi_hard_triplet':
        train_loss = losses.BatchAllTripletLoss(model=model)
    elif opt.loss_type == 'cosine_similarity':
        train_loss = losses.CosineSimilarityLoss(model=model)
    elif opt.loss_type == 'triplet':
        train_loss = losses.TripletLoss(model=model)

    # valid setting
    if 'triplet' in opt.loss_type:
        evaluator = evaluation.TripletEvaluator.from_input_examples(valid_examples)
    else:
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(valid_examples)

    train_dataloader = DataLoader(
        dataset=train_examples,
        shuffle=True,
        batch_size=opt.batch_size,
    )
    
    observer = TrainingObserver(
        limit_steps=opt.limit_steps,
        output_path=opt.output_path,
    )
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=opt.epochs,
        warmup_steps=opt.warmup_steps,
        evaluation_steps=opt.evaluation_steps,
        output_path=opt.output_path,
        save_best_model=True,
        use_amp=opt.use_amp,
        callback=observer.check_loss,
    )
    
    
if __name__ == '__main__':
    
    main()

