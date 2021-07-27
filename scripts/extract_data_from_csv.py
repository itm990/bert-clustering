import argparse
import csv
import re


def class2num(class_list):
    class_vary = sorted(set(class_list))
    class_dict = { class_:str(idx) for idx, class_ in enumerate(class_vary) }
    classes = [ class_dict[class_] for class_ in class_list ]
    return classes


def clean_sentence(sent):
        
    # remove 【tag】s
    print('[INFO] Remove tags')
    sent = re.sub('【[^【】]*】　*', '', sent)
    sent = re.sub('［[^［］]*］　*', '', sent)
    
    # hankaku to zenkaku
    print('[INFO] Convert hankaku to zenkaku')
    ZEN = "".join(chr(0xff01 + i) for i in range(94))
    HAN = "".join(chr(0x21 + i) for i in range(94))
    HAN2ZEN = str.maketrans(HAN, ZEN)
    sent = sent.translate(HAN2ZEN)
    sent = re.sub(' ', '　', sent)
    
    return sent
    

def extract_data(csv_file_list):
    
    sentence_list = []
    class_list = []
    for csv_file_names in csv_file_list:
        
        print('Open', csv_file_names)
        with open(csv_file_names) as csv_file:
            data = csv.reader(csv_file, delimiter=",",
                              doublequote=True,
                              lineterminator="\r\n",
                              quotechar='"',
                              skipinitialspace=True)
            head = next(data)
            
            for line in data:
                
                # extract sentence (from row I and J)
                sent = '{}{}'.format(line[8], line[9])
                # extract class label (from row P)
                cls  = '{}'.format(line[15])
                
                if len(sent) != 0 and len(cls) != 0:
                    sentence_list.append(sent)
                    class_list.append(cls)

    print('[INFO] Get sentences {}'.format(len(sentence_list)))
    print('[INFO] Get class labels {}'.format(len(class_list)))
    sentences = '{}\n'.format('\n'.join(sentence_list))
    class_list = class2num(class_list)
    classes = '{}\n'.format('\n'.join(class_list))
    
    return sentences, classes


def main():
            
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_from', type=str, nargs='+', default=['../data/No.1_class_0120.csv'])
    parser.add_argument('--save_sent_as', type=str, default='../data/1_0120_sent.txt')
    parser.add_argument('--save_cls_as',  type=str, default='../data/1_0120_cls.txt')
    opt = parser.parse_args()
    
    sentences, classes = extract_data(opt.load_from)
    sentences = clean_sentence(sentences)
    
    with open(opt.save_sent_as, mode='w') as save_file:
        save_file.write(sentences)
    print('[INFO] Save sentences as', opt.save_sent_as)
    
    with open(opt.save_cls_as, mode='w') as save_file:
        save_file.write(classes)
    print('[INFO] Save classes as', opt.save_cls_as)


if __name__ == '__main__':
    main()
