import argparse
import csv
import re


def class2num(class_list):
    class_vary = sorted(set(class_list))
    class_dict = { class_:str(idx) for idx, class_ in enumerate(class_vary) }
    classes = [ class_dict[class_] for class_ in class_list ]
    return classes


def list2str(lst):
    return '{}\n'.format('\n'.join(lst))


def str2list(string):
    return string.strip().split('\n')


def separate_by_tags(sent_list, max_sent_num=8):
    new_sent_list = []
    new_label_list = []
    for label, sent in enumerate(sent_list):
        # separate by '【num】 tags'
        separate_list = re.split('【[０-９]{4}】|［[０-９]{4}］', sent)
        if len(separate_list) > 1:
            new_sent_list += separate_list[:max_sent_num]
            new_label_list += [ str(label) ] * len(separate_list[:max_sent_num])
    return new_sent_list, new_label_list


def han2zen(sent):
    # hankaku to zenkaku
    ZEN = "".join(chr(0xff01 + i) for i in range(94))
    HAN = "".join(chr(0x21 + i) for i in range(94))
    HAN2ZEN = str.maketrans(HAN, ZEN)
    sent = sent.translate(HAN2ZEN)
    return sent


def remove_tags(sent):
    # remove tags
    sent = re.sub('【[^【】]*】　*', '', sent)
    sent = re.sub('［[^［］]*］　*', '', sent)
    return sent


def extract_data(csv_file_list):
    
    sentence_list = []
    for csv_file_names in csv_file_list:
        
        print('Open', csv_file_names)
        with open(csv_file_names, 'r', encoding='cp932') as csv_file:
            data = csv.reader(csv_file, delimiter=",",
                              doublequote=True,
                              lineterminator="\r\n",
                              quotechar='"',
                              skipinitialspace=True)
            head = next(data)
            
            for line in data:
                # sentence number
                if line[0].isdecimal():
                    sentence_list.append(line[2])
    print('[INFO] Get {} lines'.format(len(sentence_list)))
    sentence_list, label_list =  separate_by_tags(sentence_list)
    
    sentences = '\n'.join(sentence_list)
    sentences = han2zen(sentences)
    sentences = remove_tags(sentences)
    
    # remove null sentence
    new_sent_list = []
    new_label_list = []
    for label, sent in zip(label_list, sentences.split('\n')):
        if sent.strip() != '':
            new_sent_list.append(sent)
            new_label_list.append(label)
        
    print('[INFO] Get {} sentences'.format(len(new_sent_list)))
    print('[INFO] Get {} labels'.format(len(new_label_list)))
    sentences = list2str(new_sent_list)
    labels = list2str(new_label_list)
    
    return sentences, labels


def save_to_file(filename, string):
    with open(filename, encoding='utf-8', mode='w') as save_file:
        save_file.write(string)
    print('[INFO] Save as', filename)


def main():
            
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_from', type=str, nargs='+', default=['../data/For_Finetuning.csv'])
    parser.add_argument('--save_sent_as', type=str, default='../data/for_ft_sent.txt')
    parser.add_argument('--save_cls_as', type=str, default='../data/for_ft_cls.txt')
    opt = parser.parse_args()
    
    sentences, classes = extract_data(opt.load_from)
    
    save_to_file(opt.save_sent_as, sentences)
    save_to_file(opt.save_cls_as, classes)
    

if __name__ == '__main__':
    main()
