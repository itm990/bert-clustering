import argparse
import numpy as np
from sklearn import metrics


def purity_score(gold, pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(gold, pred)
    
    # purity
    purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    inv_purity = np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)
    f_value = 2 / ( 1/purity + 1/inv_purity )
    return purity, inv_purity, f_value


def evaluate(opt):
    
    # prepare gold data
    print('[INFO] Get gold labels from {}'.format(opt.gold_path))
    gold_list = []
    with open(opt.gold_path) as gold_file:
        for label in gold_file:
            gold_list.append(label.strip())
            
    # prepare predict data
    print('[INFO] Get predict labels from {}'.format(opt.pred_path))
    pred_list = []
    with open(opt.pred_path) as pred_file:
        for label in pred_file:
            pred_list.append(label.strip())
            
    if len(gold_list) != len(pred_list):
        print('[Error] Not match number of documents, gold : {}, predict : {}'.format(len(gold_list), len(pred_list)))
        exit()
    
    # purity
    print('[INFO] Calculate purity')
    gold = np.array(gold_list)
    pred = np.array(pred_list)
    score = purity_score(gold, pred)
    summary = 'Purity: {}\nInverse Purity: {}\nF-value: {}\n'.format(score[0], score[1], score[2])
    print(summary)
    print('[INFO] Save purity score as {}'.format(opt.output_path))
    with open(opt.output_path, mode='w') as output_file:
        output_file.write(summary)
        

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_path', type=str, default='../data/1-7_cls.txt')
    parser.add_argument('--pred_path', type=str, default='../data/1-7_kmeans_pred.txt')
    parser.add_argument('--output_path', type=str, default='../data/1-7_kmeans_purity.txt')
    opt = parser.parse_args()
    
    evaluate(opt)
    

if __name__ == '__main__':    
    main()

