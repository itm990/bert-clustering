from argparse import ArgumentParser

import torch
import torch.nn as nn

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


def load_sentence_and_label(sent_paths, cls_paths):
    print("[INFO] Load sentence data")
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
    
    print("[INFO] Got {} labels ({} types)".format(len(label_list), len(set(label_list))))
    
    return sentence_list, label_list


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = { key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


def main():
    
    parser = ArgumentParser()
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--class_num", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--train_sent", type=str, nargs="+", default=[])
    parser.add_argument("--train_cls",  type=str, nargs="+", default=[])
    parser.add_argument("--valid_sent", type=str, nargs="*", default=[])
    parser.add_argument("--valid_cls",  type=str, nargs="*", default=[])
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_gpu", action="store_true")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    do_eval = True if len(args.valid_sent) != 0 and len(args.valid_cls) != 0 else False
    
    train_sent, train_cls = load_sentence_and_label(args.train_sent, args.train_cls)
    if do_eval:
        valid_sent, valid_cls = load_sentence_and_label(args.valid_sent, args.valid_cls)
    
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.model_max_length = 512
    
    train_encodings = tokenizer(train_sent, truncation=True, padding=True)
    train_dataset = MyDataset(train_encodings, train_cls)
    
    valid_encodings = None,
    valid_dataset = None
    evaluation_strategy = "no"
    eval_steps = None
    if do_eval:
        valid_encodings = tokenizer(valid_sent, truncation=True, padding=True)
        valid_dataset = MyDataset(valid_encodings, valid_cls)
        evaluation_strategy = "steps"
        eval_steps = args.eval_steps
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=do_eval,
        num_train_epochs=args.max_epoch,
        logging_dir="{}/logs".format(args.output_dir),
        logging_steps=args.logging_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_steps=args.eval_steps,
    )
    
    config = BertConfig.from_json_file("{}/config.json".format(args.model_name_or_path))
    config.num_labels = args.class_num
    model = BertForSequenceClassification(config)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    
    trainer.train()
    
    model_to_save = bert_model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    

if __name__ == "__main__":
    main()

