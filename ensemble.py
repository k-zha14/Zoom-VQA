import os
import numpy as np
from scipy.stats import spearmanr, pearsonr


def sigmoid_rescale(score):
    # rescale by self-mean & self-std
    assert len(score) > 1, 'Expect input is a list not a single number.'
    score = (score - np.mean(score)) / np.std(score)
    score = 1 / (1 + np.exp(-score)) * 100
    return score


def main():
    # 1. load IQA&VQA preds
    with open('./results/iqa_preds_final.txt', 'r') as f:
        lines = f.read().splitlines()
    preds = np.array([float(line.split(',')[-1]) for line in lines])
    preds = sigmoid_rescale(preds)
    names = [line.split(',')[0] for line in lines]
    print('IQA min&max: {:.4f} / {:.4f}'.format(np.min(preds), np.max(preds)))

    # load VQA
    with open('./results/vqa_preds_final.txt', 'r') as f:
        lines = f.read().splitlines()
    vqa_preds = np.array([float(line.split(',')[-1]) for line in lines])

    # check preds is aligned
    assert len(preds) == len(vqa_preds), 'Dataset not aligned!'
    print('VQA min&max: {:.4f} / {:.4f}'.format(np.min(vqa_preds), np.max(vqa_preds)))

    ensemb_preds = 0.5 * preds + 0.5 * vqa_preds
    print('After ensemble, min&max: {:.4f} / {:.4f}'.format(np.min(ensemb_preds), np.max(ensemb_preds)))

    # 2. save the ensemble results
    assert len(names) == len(ensemb_preds)
    with open('./results/output.txt', 'w') as f:
        for name, pred in zip(names, ensemb_preds):
            print('{},{}'.format(name, pred), file=f)
    
    # 3. packup as zip file
    cmd = 'zip -jq ./results/{}.zip ./results/output.txt ./results/readme.txt'.format('final_submit')
    os.system(cmd)
    print('Ensemble done, the final preds is saved as \'./results/output.txt\' and packed as \'./results/final_submit.zip\'')


if __name__ == '__main__':
    main()

