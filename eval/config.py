import os

def get_cfg():

    cfg = {}

    # File Path
    cfg["FILE_PATH"] = {
        'model_gen_file': '/home/tfshen/pyproject/pcg/data/summary4eval/model_sample_gen.json',
        'processed_file':'/home/tfshen/pyproject/pcg/ui/data/dataset/done.json',
        'raw_test_file':'/home/tfshen/pyproject/pcg/data/split/test.json',    
        'sample_file': '/home/tfshen/pyproject/pcg/ui/data/dataset/test_sample.txt',
        'rank_file':'/home/tfshen/pyproject/pcg/ui/data/result/all_rank.json',
        'img_dir':'/data/tfshen/pcg_imagebase',
        'sample_img_dir': '/data/tfshen/pcg_img_sample',
        'navigation_path':'/data/tfshen/pcg_imagebase/navigation.json',
        'out_path':'/home/tfshen/pyproject/pcg/ui/data/dataset/done.json',
        'model_gen_out_path': '/home/tfshen/pyproject/pcg/data/summary4eval/model_sample_gen.json',
        'G_nlg_score_path': '/home/tfshen/pyproject/pcg/data/summary4eval/G_model_nlg_score.json',
        'I_nlg_score_path': '/home/tfshen/pyproject/pcg/data/summary4eval/I_model_nlg_score.json',
        'label_path': '/home/tfshen/pyproject/pcg/data/summary4eval/label.json',
        'context_path': '/home/tfshen/pyproject/pcg/data/summary4eval/context.json',
        # survey dir
        'survey_dir': '/home/tfshen/pyproject/pcg/ui/data/result/eval_sys',
        # ficl
        'ficl_deque_dir': '/home/tfshen/pyproject/pcg/data/summary4eval/ficl_deque',
        'ficl_done_dir': '/home/tfshen/pyproject/pcg/data/summary4eval/ficl_done',
        'ficl_result_path': '/home/tfshen/pyproject/pcg/data/summary4eval/ficl_result.json',
        # udcf
        'udcf_deque_dir': '/home/tfshen/pyproject/pcg/data/summary4eval/udcf_deque',
        'udcf_done_dir': '/home/tfshen/pyproject/pcg/data/summary4eval/udcf_done',
        'udcf_result_path': '/home/tfshen/pyproject/pcg/data/summary4eval/udcf_result.json',
        # vis logging
        'logging_path': '/home/tfshen/pyproject/pcg/eval/output/vis.log'
    }
    # Gen Path
    cfg["GEN_PATH"] = {
        'ofa_ft_gen_path#1': '/home/tfshen/pyproject/pcg/data/summary4eval/Generation/ofa.txt',
        'maria_ft_gen_path#1':'/home/tfshen/pyproject/pcg/data/summary4eval/Generation/maria.txt',
        'livebot_ft_gen_path#1':'/home/tfshen/pyproject/pcg/data/summary4eval/Generation/livebot.txt',
        'mplug-video_z-shot_gen_path#1':'/home/tfshen/pyproject/pcg/data/summary4eval/Generation/mplug-video.txt',
        'mplug-owl3_z-shot_gen_path#1':'/home/tfshen/pyproject/pcg/data/summary4eval/Generation/mplug-owl3.txt',
        'qwen2-vl_z-shot_gen_path#1':'/home/tfshen/pyproject/pcg/data/summary4eval/Generation/qwen2-vl.txt',
        'minicpm-v_z-shot_gen_path#1':'/home/tfshen/pyproject/pcg/data/summary4eval/Generation/minicpm-v.txt',
        'gpt-4o_z-shot_gen_path#1': '/home/tfshen/pyproject/pcg/data/summary4eval/Generation/gpt-4o.txt',
        # 'glm-4v-flash_z-shot_gen_path#1': '/home/tfshen/pyproject/pcg/data/summary4eval/Generation/glm-4v-flash.txt',
        # 'glm-4v-plus_z-shot_gen_path#1': '/home/tfshen/pyproject/pcg/data/summary4eval/Generation/glm-4v-plus.txt'
    }

    # Eval Path
    cfg["EVAL_PATH"] = {
        
    }

    # Option
    cfg['OPTION'] = {
        'mode': 'all',
        'ablation_model_name': '',
        'add_list': [],
    }

    # eval
    cfg['EVAL'] = {
        'rank_n': 4
    }


    return cfg

