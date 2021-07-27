import argparse


def make_cls_note(opt):
    
    # loadprepare data
    print('[INFO] Get labels from {}'.format(opt.input_file))
    label_list = []
    with open(opt.input_file) as input_file:
        for label in input_file:
            label_list.append(label.strip())
            
    # prepare predict data
    label_type = set(label_list)
    print('[INFO] Get {} label types'.format(len(label_type)))
    label_dict = {}# key : '' for key in label_type }
    for num, label in enumerate(label_list):
        doc_num = num + 1
        if label in label_dict:
            label_dict[label] += ', {}'.format(doc_num)
        else:
            label_dict[label] = str(doc_num)

    # save clustering data
    print('[INFO] Save as {}'.format(opt.output_file))
    string = ''
    for key, value in label_dict.items():
        string += 'cluster: {}\n'.format(key)
        string += 'doc_num: {}\n\n'.format(value)
    with open(opt.output_file, mode='w') as output_file:
        output_file.write(string)

    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='../data/cls.txt')
    parser.add_argument('--output_file', type=str, default='../data/cls.txt')
    opt = parser.parse_args()
    
    make_cls_note(opt)
    

if __name__ == '__main__':    
    main()

