import os
import torch
import json
import re
import nltk
import math
import jieba
import logging
from nlgeval import NLGEval
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from nltk.util import ngrams 
from rouge_chinese import Rouge
from bert_score import score
from tools import create_logger



n = NLGEval(metrics_to_omit=['METEOR', 'SPICE', 'SkipThoughtCS','EmbeddingAverageCosineSimilarity','VectorExtremaCosineSimilarity','GreedyMatchingScore'])

def write(result, out_path):

    if out_path is None:
        print("Out File is None ...")
    
    if not out_path.endswith('.json'):
        print("Out File doesn't meet json-format ...")

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


# 计算Dist
def distinct_n_sentence_level(sentence, n):
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)



def cal_nlgeval_metrics(hyp, ref, n=n, mode='word', out_path=None):
    assert mode in ['word', 'jieba'], '指定评估标准的计算模式须指定为[word]或[jieba]'
    
    result = {}
    all_caps = []
    
    global_bleu1 = 0
    global_bleu2 = 0
    global_bleu3 = 0
    global_bleu4 = 0
    global_cider = 0
    global_rouge = 0


    hyp_list = []
    ref_list = [[] for i in range(len(ref))]
    

    for sample_idx in tqdm(range(len(hyp))):

        cap = hyp[sample_idx]

        if mode == 'word':
            cap = ' '.join(cap)
            all_caps += cap.split()
        elif mode == 'jieba':
            jieba_cap = jieba.cut(cap)
            cap = ' '.join(jieba_cap)
            all_caps += cap.split()
        

        hyp_list.append(cap)


        unit_bleu1 = 0
        unit_bleu2 = 0
        unit_bleu3 = 0
        unit_bleu4 = 0
        unit_rouge = 0

        for ref_idx in range(len(ref)):

            label = ref[ref_idx][sample_idx]
 
            if mode == 'word':
                label = ' '.join(label)
            elif mode == 'jieba':
                label = ' '.join(jieba.cut(label))
    
            ref_list[ref_idx].append(label)


            try:
                score = n.compute_individual_metrics(hyp=cap, ref=[label])
            except:
                print(f'nlgeval在计算第{sample_idx}个hyp样本，第{ref_idx}个ref时出现错误，跳过该hyp-ref样本(以上计数均从0开始)')
                continue
            unit_bleu1 += score['Bleu_1']
            unit_bleu2 += score['Bleu_2']
            unit_bleu3 += score['Bleu_3']
            unit_bleu4 += score['Bleu_4']
            unit_rouge += score['ROUGE_L']


        unit_bleu1 = unit_bleu1 / len(ref)
        unit_bleu2 = unit_bleu2 / len(ref)
        unit_bleu3 = unit_bleu3 / len(ref)
        unit_bleu4 = unit_bleu4 / len(ref)
        unit_rouge = unit_rouge / len(ref)

        global_bleu1 += unit_bleu1
        global_bleu2 += unit_bleu2
        global_bleu3 += unit_bleu3
        global_bleu4 += unit_bleu4
        
        global_rouge += unit_rouge

    global_bleu1 = global_bleu1 / len(hyp)
    global_bleu2 = global_bleu2 / len(hyp)
    global_bleu3 = global_bleu3 / len(hyp)
    global_bleu4 = global_bleu4 / len(hyp)
    global_rouge = global_rouge / len(hyp)

    try:
        score4CIDEr = n.compute_metrics(ref_list=ref_list, hyp_list=hyp_list)
    except:
        raise SyntaxError('在最终计算CIDEr分数的compute_metrics函数调用时发生错误')
    global_cider = score4CIDEr['CIDEr']

    dist1 = distinct_n_sentence_level(all_caps, 1)
    dist2 = distinct_n_sentence_level(all_caps, 2)
    dist3 = distinct_n_sentence_level(all_caps, 3)

    result['Bleu_1'] = global_bleu1
    result['Bleu_2'] = global_bleu2
    result['Bleu_3'] = global_bleu3
    result['Bleu_4'] = global_bleu4
    result['CIDEr'] = global_cider
    result['ROUGE_L'] = global_rouge
    result['Dist_1'] = dist1
    result['Dist_2'] = dist2
    result['Dist_3'] = dist3

    
    if out_path is not None:
        write(result=result, out_path=out_path)
    return result


def cal_nltk_metrics(hyp, ref, mode='word', use_smooth=False, out_path=None, cal_bert_score=False):

    result = {}

    # =============== BERT SCORE ================ #
    precisions, recalls, f1 = 0, 0, 0
    if cal_bert_score == True:
        for idx in range(len(ref)):
            unit_p, unit_r, unit_f1 = score(cands=hyp, refs=ref[idx], model_type='bert-base-chinese', lang='zh', verbose=True)
            precisions += unit_p.mean()
            recalls += unit_r.mean()
            f1 += unit_f1.mean()
        precisions = round(precisions / len(ref), 3)
        recalls = round(recalls / len(ref), 3)
        f1 = round(f1 / len(ref), 3)

    result['Precisions'] = precisions
    result['Recalls'] = recalls
    result['F1'] = f1

    # ============================================ #

    if use_smooth == True:
        smooth = SmoothingFunction()
    else:
        smooth = None

    # 定义ROUGE
    rouge = Rouge()
    all_caps = []
    
    cnt = 0 
    bleu1 = 0.
    bleu2 = 0.
    bleu3 = 0.
    bleu4 = 0.
    rouge1 = 0.
    rouge2 = 0.
    rougeL = 0.
    meteor =0.
    nist1 = 0.
    nist2 = 0.
    nist3 = 0.
    nist4 = 0.
    
    for sample_idx in tqdm(range(len(hyp))):
        cap = hyp[sample_idx]

        if mode == 'word':
            cap = (' '.join(cap)).split()
            all_caps += cap
        elif mode == 'jieba':
            jieba_cap = jieba.cut(cap)
            cap = (' '.join(jieba_cap)).split()
            all_caps += cap
        for ref_idx in range(len(ref)):
            label = ref[ref_idx][sample_idx]
            if mode == 'word':
                label = (' '.join(label)).split()
            elif mode == 'jieba':
                label = (' '.join(jieba.cut(label))).split()
            min_len = min(len(cap), len(label)) 
            lens = min(min_len, 4)
            if lens == 0:
                continue
            
            temp_cap = ' '.join(cap)
            temp_label = ' '.join(label)

            try:
                rouge_score = rouge.get_scores(temp_cap, temp_label)
                rouge1 += rouge_score[0]["rouge-1"]['r']
                rouge2 += rouge_score[0]["rouge-2"]['r']
                rougeL += rouge_score[0]["rouge-l"]['f']
            
                label = [label]
                if use_smooth:
                    if lens >= 1:
                        bleu1 += sentence_bleu(label, cap, weights=(1, 0, 0, 0), smoothing_function=smooth.method1)
                        nist1 += sentence_nist(label, cap, 1)
                    if lens >= 2:
                        bleu2 += sentence_bleu(label, cap, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
                        nist2 += sentence_nist(label, cap, 2)
                    if lens >= 3:
                        bleu3 += sentence_bleu(label, cap, weights=(0.333, 0.333, 0.333, 0), smoothing_function=smooth.method1)
                        nist3 += sentence_nist(label, cap, 3)
                    if lens >= 4:
                        bleu4 += sentence_bleu(label, cap, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
                        nist4 += sentence_nist(label, cap, 4)
                else:
                    if lens >= 1:
                        bleu1 += sentence_bleu(label, cap, weights=(1, 0, 0, 0))
                        nist1 += sentence_nist(label, cap, 1)
                    if lens >= 2:
                        bleu2 += sentence_bleu(label, cap, weights=(0.5, 0.5, 0, 0))
                        nist2 += sentence_nist(label, cap, 2)
                    if lens >= 3:
                        bleu3 += sentence_bleu(label, cap, weights=(0.333, 0.333, 0.333, 0))
                        nist3 += sentence_nist(label, cap, 3)
                    if lens >= 4:
                        bleu4 += sentence_bleu(label, cap, weights=(0.25, 0.25, 0.25, 0.25))
                        nist4 += sentence_nist(label, cap, 4)
         
                cnt += 1
            except:
                print(f'nltk在计算第{sample_idx}个hyp样本，第{ref_idx}个ref时出现错误，跳过该hyp-ref样本(以上计数均从0开始)')
                continue

    dist1 = distinct_n_sentence_level(all_caps, 1)
    dist2 = distinct_n_sentence_level(all_caps, 2)
    dist3 = distinct_n_sentence_level(all_caps, 3)
    bleu1 /= cnt
    bleu2 /= cnt
    bleu3 /= cnt
    bleu4 /= cnt
    rouge1 /= cnt
    rouge2 /= cnt
    rougeL /= cnt
    # meteor /= cnt
    nist1 /= cnt
    nist2 /= cnt
    nist3 /= cnt
    nist4 /= cnt
    result['Bleu_1'] = bleu1
    result['Bleu_2'] = bleu2
    result['Bleu_3'] = bleu3
    result['Bleu_4'] = bleu4
    result['ROUGE_L'] =rougeL
    result['METEOR'] = 0 
    result['CIDEr'] = 0 
    result['Nist_1'] = nist1
    result['Nist_2'] = nist2
    result['Nist_3'] = nist3
    result['Nist_4'] = nist4
    result['Dist1'] = dist1
    result['Dist2'] = dist2
    result['Dist3'] = dist3

    if out_path is not None:
        write(result=result, out_path=out_path)
    return result



def cal_multi_metrics(hyp, ref, n=None, outdir=None, mode='test_metrcis', cal_bert_score=False):
    # 处理outdir
    if outdir is None:
        outdir_name = './metric_result'
        os.mkdir(outdir_name)
    else:
        outdir = os.path.join(outdir, mode)
        if os.path.exists(outdir):
            assert os.path.isdir(outdir), '指定路径须为文件夹路径'
        else:
            os.mkdir(outdir)

    # nlgeval-word-level
    nlg_word_name = os.path.join(outdir, 'nlg_word_level.json')
    nlg_word_result = cal_nlgeval_metrics(hyp=hyp, ref=ref, mode='word', out_path=nlg_word_name)
    # nlgeval-jieba-level
    nlg_jieba_name = os.path.join(outdir, 'nlg_jieba_level.json')
    nlg_jieba_result = cal_nlgeval_metrics(hyp=hyp, ref=ref, mode='jieba',out_path=nlg_jieba_name)
    # nltk-word-level
    nltk_word_name = os.path.join(outdir, 'nltk_word_level.json')
    nltk_word_result = cal_nltk_metrics(hyp=hyp, ref=ref, mode='word', use_smooth=False, out_path=nltk_word_name, cal_bert_score=cal_bert_score)
    # nltk-jieba-level
    nltk_jieba_name = os.path.join(outdir, 'nltk_jieba_level.json')
    nltk_jieba_result = cal_nltk_metrics(hyp=hyp, ref=ref, mode='jieba', use_smooth=False, out_path=nltk_jieba_name)
    # nltk-word_smooth-level
    nltk_word_sm_name = os.path.join(outdir, 'nltk_word_sm_level.json')
    nltk_word_sm_result = cal_nltk_metrics(hyp=hyp, ref=ref,mode='word', use_smooth=True, out_path=nltk_word_sm_name)
    # nltk-jieba-smooth-level
    nltk_jieba_sm_name = os.path.join(outdir, 'nltk_jieba_sm_level.json')
    nltk_jieba_sm_result = cal_nltk_metrics(hyp=hyp, ref=ref,mode='jieba', use_smooth=True, out_path=nltk_jieba_sm_name)
    
    print('=========nlgeval-word-level============')
    print(nlg_word_result)
    print('=======================================')
    print('=========nlgeval-jieba-level============')
    print(nlg_jieba_result)
    print('=======================================')
    print('=========nltk-word-level============')
    print(nltk_word_result)
    print('=======================================')
    print('=========nltk-jieba-level============')
    print(nltk_jieba_result)
    print('=======================================')
    print('=========nltk-word-smooth-level============')
    print(nltk_word_sm_result)
    print('=======================================')
    print('=========nltk-jieba-smooth-level============')
    print(nltk_jieba_sm_result)
    print('=======================================')
    
    return nltk_word_result



