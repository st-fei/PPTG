'''
2024.11.11 by stf
用于建立评估体系
'''

import os
import re
import json
import shutil
import logging
import numpy as np
from typing import Optional, List, Dict
from tqdm import tqdm
from config import get_cfg
from tools import load_json, text_process, create_logger
from nlgeval import NLGEval
from cal_metrics import cal_nlgeval_metrics


class NDCG_Indiv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sample_data = load_json(cfg['FILE_PATH']['processed_file'])
        self.rel_table = [3.2, 2.5, 2.1, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0] # 预定义超参
        # self.rel_table = [4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 均匀幅度下降
        # self.rel_table = [4.0, 3.0, 1.8, 0.5, 0.0, 0.0, 0.0, 0.0] # 较大梯度下降
        self.human_label = self.get_human_eval()
        self.model_list = ['ofa_ft_gen_path#1', 'maria_ft_gen_path#1', 'livebot_ft_gen_path#1', 'mplug-video_z-shot_gen_path#1','mplug-owl3_z-shot_gen_path#1', 'qwen2-vl_z-shot_gen_path#1', 'minicpm-v_z-shot_gen_path#1', 'gpt-4o_z-shot_gen_path#1']

    def load_model_res(self, ):
        '''
        加载后gen格式:{model_name: {anonymous_name: gen_title}}
        加载后label格式:{anonymous_name: title}
        加载后context格式:{anonymous_name: [history_title1, history_title2, ...]}
        '''
        # check
        if os.path.isfile(self.cfg['FILE_PATH']['model_gen_file']) and os.path.isfile(self.cfg['FILE_PATH']['label_path']) and os.path.isfile(self.cfg['FILE_PATH']['context_path']):
            gen = load_json(self.cfg['FILE_PATH']['model_gen_file'])
            label = load_json(self.cfg['FILE_PATH']['label_path'])
            context = load_json(self.cfg['FILE_PATH']['context_path'])
            return gen, label, context
        
        # load raw data
        raw_data = load_json(self.cfg['FILE_PATH']['raw_test_file'])
        raw_anonymous_list = list(raw_data.keys())
        # map anonymous_name -> line
        raw_anonymous_map = {raw_anonymous_list[i]: i for i in range(len(raw_anonymous_list))}
        # load sample data
        gen = {model_name: {} for model_name in self.cfg['GEN_PATH']} # store all models gen
        label = {} # store all models label
        context = {} # store all data context
        # ---------------------------------------- # 
        # 按组遍历采样数据文件，获取所有模型的生成结果
        cnt = 0
        for group_name, group_value in self.sample_data.items():
            for sample_name in group_value:
                # 此时sample_name为当前anonymous_name
                if sample_name != 'count':
                    # 获取标签
                    _label = text_process(raw_data[sample_name]['target']['title'])
                    label[sample_name] = _label
                    # 获取上下文
                    context[sample_name] = []
                    history_list = group_value[sample_name]['history']
                    for history_note in history_list:
                        context[sample_name].append(text_process(history_note['title']))
                        
                    # 遍历所有模型获取结果
                    for model_name, gen_path in self.cfg['GEN_PATH'].items():
                        if gen_path != '' and os.path.isfile(gen_path):
                            with open(gen_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                if len(lines) == 2039:
                                    tgt_line = raw_anonymous_map[sample_name]
                                    title = lines[tgt_line]
                                    title = text_process(title)
                                    gen[model_name][sample_name] = title
                                elif len(lines) == 400:
                                    title = lines[cnt]
                                    title = text_process(title)
                                    gen[model_name][sample_name] = title
                                else:
                                    raise ValueError(f'模型{model_name}生成文本行数:{len(lines)}不匹配')
                    cnt += 1 
        
        # 写入
        with open(self.cfg['FILE_PATH']['model_gen_out_path'], 'w', encoding='utf-8') as f:
            json.dump(gen, f, ensure_ascii=False, indent=4)

        with open(self.cfg['FILE_PATH']['label_path'], 'w', encoding='utf-8') as f:
            json.dump(label, f, ensure_ascii=False, indent=4)

        with open(self.cfg['FILE_PATH']['context_path'], 'w', encoding='utf-8') as f:
            json.dump(context, f, ensure_ascii=False, indent=4)

        return gen, label, context
        
    def calculate_nlg_score(self, gen, label):
        # 加载后gen格式:{model_name: {anonymous_name: gen_title}}
        # 加载后label格式:{anonymous_name: title}
        '''
        为每个模型计算样例nlg分数
        只有CIDEr分数需要全局文本
        nlg_score格式为:{nlg_metric: {anonymous_name: {model_name: nlg_res}}}
        '''
        # check
        if os.path.isfile(self.cfg['FILE_PATH']['I_nlg_score_path']):
            nlg_score = load_json(self.cfg['FILE_PATH']['I_nlg_score_path'])
            return nlg_score
        # initialize
        nlg_score = {
            'Bleu_1': {},
            'Bleu_2': {},
            'Bleu_3': {},
            'Bleu_4': {},
            'ROUGE_L': {},
            'Dist_1': {},
            'Dist_2': {},
            'Dist_3': {}
        }
        # calculate
        anonymous_list = list(label.keys())
        for anonymous_name in anonymous_list:
            for metric in nlg_score:
                nlg_score[metric][anonymous_name] = {}
            # start calculate
            true_title = label[anonymous_name]
            for model_name, model_value in gen.items():
                gen_title = model_value[anonymous_name]
                indiv_metric = cal_nlgeval_metrics(hyp=[gen_title], ref=[[true_title]])
                # fill in
                for metric in nlg_score:
                    nlg_score[metric][anonymous_name][model_name] = indiv_metric[metric]
        # save
        with open(self.cfg['FILE_PATH']['I_nlg_score_path'], 'w', encoding='utf-8') as f:
            json.dump(nlg_score, f, ensure_ascii=False, indent=4)
        
        return nlg_score

    def get_human_eval(self):
        '''
        return {anonymous_name: {model_name: rel(i), ...}}
        '''
        
        # 记录所有模型在不同匿名数据上的平均得分
        avg_human_scores = {}
        # 记录每个匿名数据的出现次数
        data_frequency = {}

        # 遍历每个调查问卷结果
        survey_dir = self.cfg['FILE_PATH']['survey_dir']
        for survey_prefix in os.listdir(survey_dir):
            survey_path = os.path.join(survey_dir, survey_prefix)
            survey_data = load_json(survey_path)
            for anonymous_name in survey_data:
                if anonymous_name not in data_frequency:
                    data_frequency[anonymous_name] = {'count': 0} # count记录该匿名数据在调查问卷中出现次数
                    avg_human_scores[anonymous_name] = {}
                for rank_prefix in survey_data[anonymous_name]:
                    if rank_prefix != 'tag':
                        rank = int(rank_prefix.split('_')[-1]) - 1
                        rank_model = survey_data[anonymous_name][rank_prefix]["model_name"]
                        if rank_model != 'NULL':
                            avg_human_scores[anonymous_name][rank_model] = avg_human_scores[anonymous_name].get(rank_model, 0.0) + self.rel_table[rank]
                data_frequency[anonymous_name]['count'] = data_frequency[anonymous_name]['count'] + 1

        # avg
        for anonymous_name in avg_human_scores:
            freq = data_frequency[anonymous_name]['count']
            for model_type in avg_human_scores[anonymous_name]:
                avg_human_scores[anonymous_name][model_type] = avg_human_scores[anonymous_name][model_type] / freq

        return avg_human_scores

    def stat_survey_info(self):
        '''
        用于统计问卷组信息
        '''
        group_count = {f'Group_{group_id}': 0 for group_id in range(1, 21)}
        assert os.path.isdir(self.cfg['FILE_PATH']['survey_dir']), f"填写的问卷收集文件夹:{self.cfg['FILE_PATH']['survey_dir']}不符合要求"
        survey_dir = self.cfg['FILE_PATH']['survey_dir']
        survey_list = os.listdir(survey_dir)
        for survey_file in survey_list:
            survey_path = os.path.join(survey_dir, survey_file)
            survey_data = load_json(survey_path)
            typical_anonymous_name = list(survey_data.keys())[0]
            # search group
            sample_data = load_json(self.cfg['FILE_PATH']['processed_file'])
            for group_name, group_value in sample_data.items():
                for anonymous_name in group_value:
                    if anonymous_name == typical_anonymous_name:
                        group_count[group_name] = group_count[group_name] + 1
        return group_count

    def ndcg(self, golden, current, n = -1):
        '''
        计算所有样例的ndcg平均分数
        golden: [[rel1, rel2, ...], ...]
        current: [[rel1, rel2, ...], ...]
        golden中的相关性分数会自动sort，只需填入所有相关性分数即可
        current中的相关性分数按当前评估指标结果进行排序
        '''
        # log2 discount table
        log2_table = np.log2(np.arange(2, 102)) 
        # 计算DCG@n
        def dcg_at_n(rel, n):
            rel = np.asfarray(rel)[:n] # 转numpy float数组，并取前n个值
            # 实现dcg = Sum[i=1 -> n]{2 ** rel(i) - 1} / log(i + 1)
            dcg = np.sum(np.divide(np.power(2, rel) - 1, log2_table[:rel.shape[0]]))
            return dcg
        # 共len(current)个样例来计算ndcg分数
        ndcgs = []
        for i in range(len(current)):
            # 如果规定了n，就计算ndcg@n；如果没有，就计算ndcg@len(current[i])
            k = len(current[i]) if n == -1 else n 
            # 计算idcg@k
            idcg = dcg_at_n(sorted(golden[i], reverse=True), n=k)
            # 计算dcg@k
            dcg = dcg_at_n(current[i], n=k) 
            tmp_ndcg = 0 if idcg == 0 else dcg / idcg # 计算当前搜索结果的ndcg@k
            ndcgs.append(tmp_ndcg)
        # 计算所有人工评估结果的ndcg平均值
        return 0. if len(ndcgs) == 0 else sum(ndcgs) / (len(ndcgs))

    def calculate_ndcg4onemetric(self, model_score: dict, label_score: dict, designation: Optional[list] = None):
        '''
        单独为某一个指标计算ndcg分数
        model_score: {anonymous_name: {model_name: model_res}}
        label_score: {anonymous_name: {model_name: model_res}}
        '''
        # check data num
        # assert len(model_score) == len(label_score), f'采样指标评估数量:{len(model_score)}与标签数量:{len(label_score)}不相同'
        
        # return data sample's ndcg
        if designation is not None:
            designation_result = {}

            for anonymous_name in designation:
                label = label_score[anonymous_name]
                if anonymous_name not in model_score:
                    designation_result[anonymous_name] = 0.0
                    continue
                anonymous_data = model_score[anonymous_name]
                # fill non-rank model in label
                for model_name in anonymous_data:
                    if model_name not in label:
                        label[model_name] = 0.0
                # get diff-models' rel_score
                label_rank_list = sorted(label, key=label.get, reverse=True)
                metric_rank_list = sorted(model_score[anonymous_name], key=model_score[anonymous_name].get, reverse=True)
                # level-tree
                level_tree = {} # {level: {score: xx, freq: xx}}
                level = 0
                # build level-tree
                for i, model_name in enumerate(metric_rank_list):
                    score = anonymous_data[model_name]
                    # if equal, then degrade
                    if i == 0:
                        level_tree[str(level)] = {'score': score, 'freq': 1}
                    else:
                        assert score <= level_tree[str(level)]['score'], f'invalid sort.'
                        if score == level_tree[str(level)]['score']:
                            level_tree[str(level)]['freq'] = level_tree[str(level)]['freq'] + 1
                        else:
                            level = level + 1
                            level_tree[str(level)] = {'score': score, 'freq': 1}
                golden = [[label[model_name] for model_name in label_rank_list]]
                # get current
                current = []
                for model_name in metric_rank_list:
                    human_score = label[model_name]
                    # current model's score
                    score = anonymous_data[model_name]
                    # check tree
                    for level in level_tree:
                        if score == level_tree[level]['score']:
                            if score != 0.0:
                                # current_score = human_score * (1.15 - level_tree[level]['freq'] * 0.15)
                                current_score = human_score
                            else:
                                current_score = 0.0
                            break
                    current.append(current_score)
                current = [current]
                designation_result[anonymous_name] = self.ndcg(golden=golden, current=current, n=self.cfg['EVAL']['rank_n'])
            return designation_result
        
        else:
            # Global golden & current
            G_golden = []
            G_current = []

            # iter
            for anonymous_name, anonymous_data in model_score.items():

                label = label_score[anonymous_name].copy()
                anonymous_data = anonymous_data.copy()
                # fill non-rank model in label
                for model_name in self.model_list:
                    if model_name not in label:
                        label[model_name] = 0.0
                    if model_name not in anonymous_data:
                        anonymous_data[model_name] = 0.0
                # get diff-models' rel_score
                label_rank_list = sorted(label, key=label.get, reverse=True)
                metric_rank_list = sorted(anonymous_data, key=anonymous_data.get, reverse=True)
                # level-tree
                level_tree = {} # {level: {score: xx, freq: xx}}
                level = 0
                # build level-tree
                for i, model_name in enumerate(metric_rank_list):
                    score = anonymous_data[model_name]
                    # if equal, then degrade
                    if i == 0:
                        level_tree[str(level)] = {'score': score, 'freq': 1}
                    else:
                        assert score <= level_tree[str(level)]['score'], f'invalid sort.'
                        if score == level_tree[str(level)]['score']:
                            level_tree[str(level)]['freq'] = level_tree[str(level)]['freq'] + 1
                        else:
                            level = level + 1
                            level_tree[str(level)] = {'score': score, 'freq': 1}

                # get golden
                I_golden = [label[model_name] for model_name in label_rank_list]


                # get current
                I_current = []
                for model_name in metric_rank_list:
                    human_score = label[model_name]
                    # current model's score
                    score = anonymous_data[model_name]
                    # check tree
                    for level in level_tree:
                        if score == level_tree[level]['score']:
                            if score != 0.0:
                                current_score = human_score * (1.15 - level_tree[level]['freq'] * 0.15)
                                # current_score = human_score 
                            else:
                                current_score = 0.0 # 禁止模型按照字母排序而混分 
                            break
                    I_current.append(current_score)

                # Indiv -> Global
                G_golden.append(I_golden)
                G_current.append(I_current)

            return self.ndcg(golden=G_golden, current=G_current, n=self.cfg['EVAL']['rank_n'])

    def ndcg4metric(self, nlg_score: dict, ficl_score: Optional[Dict] = None, pcg_score: Optional[Dict] = None, designation: Optional[list] = None):
        '''
        为每个评估指标计算ndcg分数
        nlg_score: {nlg_metric: {anonymous_name: {model_name: nlg_res}}}
        ficl_score: {ficl_metric: {anonymous_name: {model_name: ficl_res}}}
        '''
        metric_ndcg = {}
        # nlg_score {nlg_metric: {anonymous_name: {model_name: nlg_res}}}
        
        for nlg_metric in nlg_score:
            metric_ndcg[nlg_metric] = self.calculate_ndcg4onemetric(nlg_score[nlg_metric], self.human_label, designation=designation)
        
        # ficl_score {ficl_metric: {anonymous_name: {model_name: ficl_res}}}
        if ficl_score is not None:
            for ficl_type in ficl_score:
                metric_ndcg[ficl_type] = self.calculate_ndcg4onemetric(ficl_score[ficl_type], self.human_label, designation=designation)
        # pcg_score
        if pcg_score is not None:
            for pcg_metric in pcg_score:
                print(f'{pcg_metric}: {len(pcg_score[pcg_metric])}/400')
                metric_ndcg[pcg_metric] = self.calculate_ndcg4onemetric(pcg_score[pcg_metric], self.human_label, designation=designation)
        return metric_ndcg
        
    def parse_ficl_args(self, ficl_name):
        # back to forward parse
        if not ficl_name.endswith('.json'):
            raise ValueError(f'ficl name: {ficl_name} does not meet json-format requirement')
        ficl_name = ficl_name[:-5]
        ficl_unique_name, unique_id = ficl_name.split('#')[0], ficl_name.split('#')[-1]
        auto_rater_model = ficl_unique_name[:-8]
        ficl_args = ficl_unique_name[-7:]
        split_ficl_args = ficl_args.split('-')
        ficl_example_num, use_ficl_history_title, use_ficl_history_img, use_ficl_target_img = split_ficl_args[0], split_ficl_args[1], split_ficl_args[2], split_ficl_args[3]
        return {
            'auto_rater_model': auto_rater_model,
            'ficl_example_num': ficl_example_num,
            'use_ficl_history_title': use_ficl_history_title,
            'use_ficl_history_img': use_ficl_history_img,
            'use_ficl_target_img': use_ficl_target_img,
            'unique_id': unique_id
        }       

    def parse_ficl_file(self, file_path: str):
        '''
        解析并提取Emotion、Style、Relevance分数
        '''
        result = {}
        # 正则表达式匹配 Emotion, Style, Relevance 的分数
        # TODO 不同模型输出格式有所不同，需要根据模型名调整
        emotion_pattern = re.compile(r"Emotion(?: similarity)?:\s*([\d.]+)")
        style_pattern = re.compile(r"(?:Language )?Style(?: similarity)?:\s*([\d.]+)")
        relevance_pattern = re.compile(r"(?:Content )?Relevance(?: similarity)?:\s*([\d.]+)")
        ficl_data = load_json(file_path)
        for anonymous_name in tqdm(ficl_data):
            rater_data = ficl_data[anonymous_name]
            if anonymous_name not in result:
                result[anonymous_name] = {}
            for model_name, eval_response in rater_data.items():
                # 初始化
                emotion_score = None
                style_score = None
                relevance_score = None
                # 正则表达式匹配
                emotion_match = emotion_pattern.search(eval_response)
                style_match = style_pattern.search(eval_response)
                relevance_match = relevance_pattern.search(eval_response)
                # 提取
                # 如果找到匹配项则提取分数并清理字符串
                if emotion_match:
                    try:
                        emotion_score = float(emotion_match.group(1).strip('.'))
                    except ValueError:
                        emotion_score = None
                if style_match:
                    try:
                        style_score = float(style_match.group(1).strip('.'))
                    except ValueError:
                        style_score = None
                if relevance_match:
                    try:
                        relevance_score = float(relevance_match.group(1).strip('.'))
                    except ValueError:
                        relevance_score = None
                # 存储
                result[anonymous_name][model_name] = {
                    'Emotion': emotion_score,
                    'Style': style_score,
                    'Relevance': relevance_score
                }
        return result

    def parse_ficl_score(self, ficl_result: dict, angle: str):
        '''
        ficl_result: {anonymous_name: {model_name: {emotion: xx, style: xx}}}
        return: {anonymous_name: {model_name: model_res}}
        '''
        assert angle in ['Emotion', 'Style', 'Relevance']
        
        result = {}

        # iter anonymous data
        for anonymous_name, model_perf in ficl_result.items():
            # initialize
            result[anonymous_name] = {}
            # iter model types
            for model_name, model_result in model_perf.items():
                score = model_result[angle]
                if isinstance(score, int) or isinstance(score, float):
                    result[anonymous_name][model_name] = result[anonymous_name].get(model_name, 0.0) + score
                elif isinstance(score, str):
                    try:
                        true_score = eval(score)
                        result[anonymous_name][model_name] = result[anonymous_name].get(model_name, 0.0) + true_score
                    except:
                        result[anonymous_name][model_name] = 0.0
                elif score is None:
                    result[anonymous_name][model_name] = 0.0
                    
        return result
   
    def get_ficl_metric(self, ):
    
        '''
        return {ficl_identity: {anonymous_name: {model_name: ficl_score}}}
        '''

        # check ficl_score_path
        if os.path.exists(self.cfg['FILE_PATH']['ficl_result_path']):
            ficl_result = load_json(self.cfg['FILE_PATH']['ficl_result_path'])
        else:
            ficl_result = {}

        # check ficl_deque
        ficl_deque_dir = self.cfg['FILE_PATH']['ficl_deque_dir']
        deque_list = os.listdir(ficl_deque_dir)
        if len(deque_list) > 0:
            assert os.path.isdir(self.cfg['FILE_PATH']['ficl_done_dir']), f"Provided ficl done dir: {self.cfg['FILE_PATH']['ficl_done_dir']} does not meet dir requirement."
            ficl_done_dir = self.cfg['FILE_PATH']['ficl_done_dir'] 
            # process ficl deque data
            for ficl_file in deque_list:
                file_info = self.parse_ficl_args(ficl_name=ficl_file)
                auto_rater_model = file_info['auto_rater_model']
                if auto_rater_model not in ficl_result:
                    ficl_result[auto_rater_model] = {}
                identity_name = f"{file_info['ficl_example_num']}-{file_info['use_ficl_history_title']}-{file_info['use_ficl_history_img']}-{file_info['use_ficl_target_img']}#{file_info['unique_id']}"
                ficl_result[auto_rater_model][identity_name] = self.parse_ficl_file(file_path=os.path.join(ficl_deque_dir, ficl_file))

            # update ficl_score_path
            with open(self.cfg['FILE_PATH']['ficl_result_path'], 'w', encoding='utf-8') as f:
                json.dump(ficl_result, f, ensure_ascii=False, indent=4)
            
            # move to ficl_done dir
            for ficl_file in deque_list:
                deque_file_path = os.path.join(ficl_deque_dir, ficl_file)
                done_file_path = os.path.join(ficl_done_dir, ficl_file)
                shutil.move(deque_file_path, done_file_path)


        # calculate ficl_score
        ficl_score = {}
        for auto_rater_model in ficl_result:
            model_diff_identity = ficl_result[auto_rater_model]
            for identity_name in model_diff_identity:
                for angle in ['Emotion', 'Style', 'Relevance']:
                    identity_score = self.parse_ficl_score(model_diff_identity[identity_name], angle=angle)
                    ficl_score[f'FICL_{auto_rater_model}_{identity_name}_{angle}'] = identity_score
        return ficl_score


    def parse_udcf_args(self, udcf_name):
        '''
        解析udcf参数
        '''
        # back to forward parse
        if not udcf_name.endswith('.json'):
            raise ValueError(f'udcf name: {udcf_name} does not meet json-format requirement')
        udcf_name = udcf_name[:-5]
        udcf_unique_name, unique_id = udcf_name.split('#')[0], udcf_name.split('#')[-1]
        auto_rater_model = udcf_unique_name[:-6]
        udcf_args = udcf_unique_name[-5:]
        split_udcf_args = udcf_args.split('-')
        udcf_example_num, use_udcf_history_img, enable_udcf_streamline = split_udcf_args[0], split_udcf_args[1], split_udcf_args[2]
        return {
            'auto_rater_model': auto_rater_model,
            'udcf_example_num': udcf_example_num,
            'use_udcf_history_img': use_udcf_history_img,
            'enable_udcf_streamline': enable_udcf_streamline,
            'unique_id': unique_id
        }   


    def parse_udcf_file(self, file_path: str):
        '''
        file -> {anonymous_name: {Rankings: xxx, Reasons: xxx}}
        '''
        file_data = load_json(file_path)
        extracted_data = {}
        for key, value in file_data.items():
            # 使用正则表达式提取 Rankings 和 Reasons
            rankings = re.search(r"Rankings: (.*?), Reasons:", value, re.DOTALL)
            reasons = re.search(r"Reasons:(.*)", value, re.DOTALL)
            if rankings and reasons:
                extracted_data[key] = {
                    "Rankings": rankings.group(1).strip(),
                    "Reasons": reasons.group(1).strip(),
                }
            if rankings:
                extracted_data[key] = {
                    "Rankings": rankings.group(1).strip(),
                }
            # else:
                # raise ValueError(f'{key} response error. ')
        return extracted_data

    def parse_udcf_score(self, udcf_result: dict):
        '''
        result -> score
        udcf_result: {anonymous_name: {Rankings: xxx, Reasons: xxx}}
        return: {anonymous_name: {model_name: score}}
        '''

        # extract order func
        def parse_ranking(ranking_str):
            ranking_str = ranking_str.strip()
            split_ranking = re.split(r"( > |= )", ranking_str)
            # 初始化变量
            order = []
            current_group = []
            
            # 遍历分割后的元素
            for item in split_ranking:
                item = item.strip()
                if item == ">":  # 遇到 `>` 表示新分组
                    if current_group:
                        order.append(current_group)
                        current_group = []
                elif item == "=":  # 遇到 `=` 表示同一分组
                    continue
                else:
                    current_group.append(item)
            
            # 添加最后一组
            if current_group:
                order.append(current_group)
            
            return order

        # model_map
        model_map = {
            'model_A': 'ofa_ft_gen_path#1',
            'model_B': 'maria_ft_gen_path#1',
            'model_C': 'livebot_ft_gen_path#1',
            'model_D': 'mplug-video_z-shot_gen_path#1',
            'model_E': 'mplug-owl3_z-shot_gen_path#1',
            'model_F': 'qwen2-vl_z-shot_gen_path#1',
            'model_G': 'minicpm-v_z-shot_gen_path#1',
            'model_H': 'gpt-4o_z-shot_gen_path#1'
        }

        score = {}
        for anonymous_name in udcf_result:
            score[anonymous_name] = {}
            ranking_str = udcf_result[anonymous_name]['Rankings']
            ranking_list = parse_ranking(ranking_str=ranking_str)
            for i, ranking_item in enumerate(ranking_list):
                for sub_model in ranking_item:
                    if sub_model in model_map:
                        model_name = model_map[sub_model]
                        score[anonymous_name][model_name] = self.rel_table[i]
        return score
        
            


    def get_udcf_metric(self):
        '''
        return {udcf_type: {anonymous_name: {model_name: udcf_score}}}
        '''

        # check udcf_result_path
        if os.path.exists(self.cfg['FILE_PATH']['udcf_result_path']):
            udcf_result = load_json(self.cfg['FILE_PATH']['udcf_result_path'])
        else:
            udcf_result = {}

        # check udcf_deque
        udcf_deque_dir = self.cfg['FILE_PATH']['udcf_deque_dir']
        deque_list = os.listdir(udcf_deque_dir)
        if len(deque_list) > 0:
            assert os.path.isdir(self.cfg['FILE_PATH']['udcf_done_dir']), f"Provided udcf done dir: {self.cfg['FILE_PATH']['udcf_done_dir']} does not meet dir requirement."
            udcf_done_dir = self.cfg['FILE_PATH']['udcf_done_dir'] 
            # process udcf deque data
            for udcf_file in deque_list:
                udcf_info = self.parse_udcf_args(udcf_name=udcf_file)
                auto_rater_model = udcf_info['auto_rater_model']
                if auto_rater_model not in udcf_result:
                    udcf_result[auto_rater_model] = {}
                identity_name = f"{udcf_info['udcf_example_num']}-{udcf_info['use_udcf_history_img']}-{udcf_info['enable_udcf_streamline']}#{udcf_info['unique_id']}"
                udcf_result[auto_rater_model][identity_name] = self.parse_udcf_file(os.path.join(udcf_deque_dir, udcf_file))

            # update udcf result
            with open(self.cfg['FILE_PATH']['udcf_result_path'], 'w', encoding='utf-8') as f:
                json.dump(udcf_result, f, ensure_ascii=False, indent=4)
            
            # move to udcf_done dir
            for udcf_file in deque_list:
                deque_file_path = os.path.join(udcf_deque_dir, udcf_file)
                done_file_path = os.path.join(udcf_done_dir, udcf_file)
                shutil.move(deque_file_path, done_file_path)


        # calculate udcf_score
        udcf_score = {}
        for auto_rater_model in udcf_result:
            for identity_name in udcf_result[auto_rater_model]:
                udcf_score[f'UDCF_{auto_rater_model}-{identity_name}'] = self.parse_udcf_score(udcf_result[auto_rater_model][identity_name]) 
        return udcf_score



    def get_pcg_metric(self, gen, label, context):
        '''
        gen: {model_name: {anonymous_name: gen_title}}
        label: {anonymous_name: label_title}
        context: {anonymous_name: [history_title1, history_title2, ...]}
        return {pcg_type: {anonymous_name: {model_name: score}}}
        '''
        pcg_metric = {
            'Nlg_RR': {},
            'Nlg_Sum': {},
        }

        # traditional metric
        
        model_list = list(gen.keys())
        anonymous_list = list(label.keys())
        for anonymous_name in anonymous_list:
            label_title = label[anonymous_name]
            context_titles = context[anonymous_name]
            for metric in pcg_metric:
                pcg_metric[metric][anonymous_name] = {}
            for model_name in model_list:
                gen_title = gen[model_name][anonymous_name]
                bleu_in_gen_context = []
                bleu_in_label_context = []
                bleu_in_gen_label = 0.0
                for context_title in context_titles:
                    tmp_1 = cal_nlgeval_metrics(hyp=[gen_title], ref=[[context_title]])['Bleu_2']
                    tmp_2 = cal_nlgeval_metrics(hyp=[label_title], ref=[[context_title]])['Bleu_2']
                    bleu_in_gen_context.append(tmp_1)
                    bleu_in_label_context.append(tmp_2)
                bleu_in_gen_label = cal_nlgeval_metrics(hyp=[gen_title], ref=[[label_title]])['Bleu_2']
                bleu_in_gen_context = sum(bleu_in_gen_context) / len(bleu_in_gen_context)
                bleu_in_label_context = sum(bleu_in_label_context) / len(bleu_in_label_context)

                # Nlg_RR
                nlg_rr = bleu_in_gen_context / bleu_in_gen_label if (bleu_in_gen_label - 0.0) > 1e-6 else 0.0

                # Nlg_Sum
                nlg_threshold = 0.06
                punish_threshold = 0.20
                alpha = 0.85
                beta = 0.15
                control_factor = 0.0
        
                # 生成内容与历史重复但标签与历史并不重复，则施加惩罚
                if bleu_in_gen_context >= punish_threshold and bleu_in_label_context < nlg_threshold:
                    control_factor = - 0.15
                    nlg_sum = (alpha + control_factor) * bleu_in_gen_label - (beta - control_factor) * bleu_in_gen_context
                else:
                    control_factor = 0.15
                    nlg_sum = (alpha + control_factor) * bleu_in_gen_label - (beta - control_factor) * bleu_in_gen_context
                pcg_metric['Nlg_RR'][anonymous_name][model_name] = nlg_rr
                pcg_metric['Nlg_Sum'][anonymous_name][model_name] = nlg_sum
            
        # auto metric (udcf)
        udcf_metric = self.get_udcf_metric()
        pcg_metric.update(udcf_metric)
        

        
        return pcg_metric

    def get_global_metric(self, nlg_score, ficl_score, pcg_score, anonymous_list):
        '''
        计算平均分数
        return {metric: {model_type: score}}
        '''

        global_score = {**nlg_score, **ficl_score, **pcg_score}
        global_metric = {}
        for metric in global_score:
            global_metric[metric] = {}
            avg_score = 0.0
            for model_type in self.model_list:
                for anonymous_name in anonymous_list:
                    try:
                        avg_score += global_score[metric][anonymous_name][model_type]
                    except:
                        avg_score += 0.0
                avg_score = avg_score / len(anonymous_list)
                global_metric[metric][model_type] = avg_score
                
        return global_metric


    def run(self, ):
        '''
        计算所有采样数据的评估指标NDCG结果
        '''
        gen, label, context = self.load_model_res()
        # nlg
        nlg_score = self.calculate_nlg_score(gen=gen, label=label)
        # ficl
        ficl_score = self.get_ficl_metric()
        # pcg
        pcg_score = self.get_pcg_metric(gen, label, context)
        # global score
        global_metric = self.get_global_metric(nlg_score=nlg_score, ficl_score=ficl_score, pcg_score=pcg_score, anonymous_list=list(label.keys()))
        with open("/home/tfshen/pyproject/pcg/eval/output/metrics/global_metric.json", 'w', encoding='utf-8') as f:
            json.dump(global_metric, f, ensure_ascii=False, indent=4)
        # every metric
        metric_ndcg = self.ndcg4metric(nlg_score=nlg_score, ficl_score=ficl_score, pcg_score=pcg_score)
        with open("/home/tfshen/pyproject/pcg/eval/output/metrics/ndcg_metric.json", 'w', encoding='utf-8') as f:
            json.dump(metric_ndcg, f, ensure_ascii=False, indent=4)

        
if __name__ == "__main__":
    global logger
    cfg = get_cfg()
    logger = create_logger(log_path=cfg['FILE_PATH']['logging_path'])
    Indiv_framework = NDCG_Indiv(cfg=cfg)
    # anonymous_list = ['2K3I8P', '4H3H0F', '4T5C0J', '7O8W6H', '7P9U0F', '8D6K9T']
    # Indiv_framework.run_designation(anonymous_list=anonymous_list)
    # Indiv_framework.run()
    Indiv_framework.run()
