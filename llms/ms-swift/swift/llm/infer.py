# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
import re
import shutil
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig, GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.utils import is_torch_npu_available

from swift.tuners import Swift
from swift.utils import (append_to_jsonl, get_logger, get_main, get_model_info, read_multi_line, seed_everything,
                         show_layers)
from .utils import (DeployArguments, InferArguments, MediaTag, Template, get_additional_saved_files, get_dataset,
                    get_model_tokenizer, get_template, inference, inference_stream, is_adapter, is_quant_model,
                    sample_dataset, set_generation_config)

logger = get_logger()


# 用于加载json数据
def load_json(data_path):
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
            return data
    except:
        raise FileNotFoundError(f'Fail to load the data file {data_path}, please check if the data file exists')


def save_checkpoint(model: Optional[PreTrainedModel],
                    tokenizer: PreTrainedTokenizerBase,
                    model_cache_dir: str,
                    ckpt_dir: Optional[str],
                    target_dir: str,
                    *,
                    save_safetensors: bool = True,
                    sft_args_kwargs: Optional[Dict[str, Any]] = None,
                    **kwargs) -> None:
    if sft_args_kwargs is None:
        sft_args_kwargs = {}
    if model is not None:
        model.save_pretrained(target_dir, safe_serialization=save_safetensors)
    if hasattr(tokenizer, 'processor'):
        tokenizer.processor.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)
    model_type = getattr(tokenizer, 'model_type')
    fname_list = ['generation_config.json', 'preprocessor_config.json']
    if model_type is not None:
        fname_list += get_additional_saved_files(model_type)

    for fname in fname_list:
        tgt_path = os.path.join(target_dir, fname)
        for model_dir in [ckpt_dir, model_cache_dir]:
            if model_dir is None:
                continue
            src_path = os.path.join(model_dir, fname)
            if os.path.isfile(src_path):
                shutil.copy(src_path, tgt_path)
                break
            elif os.path.isdir(src_path):
                shutil.copytree(src_path, tgt_path)
                break
    # configuration.json
    configuration_fname = 'configuration.json'
    new_configuration_path = os.path.join(target_dir, configuration_fname)
    for model_dir in [ckpt_dir, model_cache_dir]:
        if model_dir is None:
            continue
        old_configuration_path = os.path.join(model_dir, configuration_fname)
        if os.path.exists(old_configuration_path):
            with open(old_configuration_path, 'r', encoding='utf-8') as f:
                res = json.load(f)
            res.pop('adapter_cfg', None)
            with open(new_configuration_path, 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=4)
            break
    if ckpt_dir is not None:
        # sft_args.json
        sft_args_fname = 'sft_args.json'
        old_sft_args_path = os.path.join(ckpt_dir, sft_args_fname)
        new_sft_args_path = os.path.join(target_dir, sft_args_fname)
        if os.path.exists(old_sft_args_path):
            with open(old_sft_args_path, 'r', encoding='utf-8') as f:
                res = json.load(f)
            res['sft_type'] = 'full'
            for k in ['dtype', 'quant_method']:
                v = sft_args_kwargs.get(k)
                if v is not None:
                    res[k] = v
            with open(new_sft_args_path, 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
        # value head weights
        value_head_weights_fname_list = ['value_head.safetensors', 'value_head.bin']
        for fname in value_head_weights_fname_list:
            old_value_head_weights_path = os.path.join(ckpt_dir, fname)
            new_value_head_weights_path = os.path.join(target_dir, fname)
            if os.path.exists(old_value_head_weights_path):
                shutil.copy(old_value_head_weights_path, new_value_head_weights_path)

# 将LoRA（低秩适配矩阵）权重合并到模型检查点中，并将合并后的权重保存到新的目录
def merge_lora(args: InferArguments,
               replace_if_exists=False,
               device_map: Optional[str] = None,
               **kwargs) -> Optional[str]:
    logger.info(f'replace_if_exists: {replace_if_exists}')
    # 确保检查点目录不为None
    assert args.ckpt_dir is not None, 'args.ckpt_dir is not specified.'
    assert args.sft_type in ('lora', 'adalora', 'longlora', 'llamapro'), 'Only supports lora & llamapro series models'
    assert not is_quant_model(
        args.model_type), f'{args.model_type} is a quantized model and does not support merge-lora.'
    if args.quantization_bit != 0:
        logger.warning('It is not recommended to merge quantized models, '
                       'as this can result in performance degradation')
    ckpt_dir, ckpt_name = os.path.split(args.ckpt_dir)
    # 构造合并后LoRA权重的保存路径merged_lora_path
    merged_lora_path = os.path.join(ckpt_dir, f'{ckpt_name}-merged')
    logger.info(f'merged_lora_path: `{merged_lora_path}`')
    # 合并LoRA权重
    if os.path.exists(merged_lora_path) and not replace_if_exists:
        logger.info(f'The weight directory for the merged LoRA already exists in {args.ckpt_dir}, '
                    'skipping the saving process. '
                    'you can pass `replace_if_exists=True` to overwrite it.')
    else:
        if device_map is None:
            device_map = args.merge_device_map
        logger.info(f'merge_device_map: {device_map}')
        model, template = prepare_model_template(args, device_map=device_map, task='export')
        logger.info('Merge LoRA...')
        Swift.merge_and_unload(model)
        model = model.model
        logger.info('Saving merged weights...')
        save_checkpoint(
            model,
            template.tokenizer,
            model.model_dir,
            args.ckpt_dir,
            merged_lora_path,
            save_safetensors=args.save_safetensors,
            sft_args_kwargs={'dtype': args.dtype})
        logger.info(f'Successfully merged LoRA and saved in {merged_lora_path}.')
    logger.info("Setting args.sft_type: 'full'")
    logger.info(f'Setting args.ckpt_dir: {merged_lora_path}')
    args.sft_type = 'full'
    args.ckpt_dir = merged_lora_path
    return merged_lora_path


def prepare_model_template(args: InferArguments,
                           *,
                           device_map: Optional[str] = None,
                           task: Literal['infer', 'export'] = 'infer',
                           automodel_class=None) -> Tuple[PreTrainedModel, Template]:
    from .sft import get_default_device_map
    if is_torch_npu_available():
        print(f'device_count: {torch.npu.device_count()}')
    else:
        print(f'device_count: {torch.cuda.device_count()}')
    model_kwargs = {}
    if device_map is None:
        device_map = get_default_device_map()
    model_kwargs['device_map'] = device_map
    if device_map == 'auto':
        model_kwargs['low_cpu_mem_usage'] = True
    model_kwargs['device_map'] = device_map
    if args.device_max_memory:
        assert len(args.device_max_memory) == torch.cuda.device_count()
        model_kwargs['max_memory'] = {i: mem for i, mem in enumerate(args.device_max_memory)}

    # Loading Model and Tokenizer
    if hasattr(args, 'quant_config'):
        model_kwargs['quantization_config'] = args.quant_config
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            args.load_in_8bit,
            args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant)
        if args.bnb_4bit_compute_dtype is None:
            quantization_config.bnb_4bit_compute_dtype = None
        logger.info(f'quantization_config: {quantization_config.__dict__}')
        model_kwargs['quantization_config'] = quantization_config
    kwargs = {}
    if args.use_flash_attn is not None:
        kwargs['use_flash_attn'] = args.use_flash_attn
    model_id_or_path = None
    if args.sft_type == 'full' and args.ckpt_dir is not None:
        model_id_or_path = args.ckpt_dir
    elif args.model_id_or_path is not None:
        model_id_or_path = args.model_id_or_path
    if automodel_class is not None:
        kwargs['automodel_class'] = automodel_class
    if args.local_repo_path:
        kwargs['local_repo_path'] = args.local_repo_path
    if args.rope_scaling:
        kwargs['rope_scaling'] = args.rope_scaling
        kwargs['max_length'] = args.max_length
    model, tokenizer = get_model_tokenizer(
        args.model_type,
        args.torch_dtype,
        model_kwargs,
        model_id_or_path=model_id_or_path,
        revision=args.model_revision,
        quant_method=args.quant_method,
        **kwargs)

    if model.max_model_len is None:
        model.max_model_len = args.max_model_len
    elif args.max_model_len is not None:
        if args.max_model_len <= model.max_model_len:
            model.max_model_len = args.max_model_len
        else:
            raise ValueError('args.max_model_len exceeds the maximum max_model_len supported by the model.'
                             f'args.max_model_len: {args.max_model_len}, model.max_model_len: {model.max_model_len}')
    if task == 'infer':
        logger.info(f'model_config: {model.config}')
        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
        model._generation_config_origin = model.generation_config
        set_generation_config(model, generation_config)
        logger.info(f'model.generation_config: {model.generation_config}')

        if model.generation_config.num_beams != 1:
            args.stream = False
            logger.info('Setting args.stream: False')

    # Preparing LoRA
    if is_adapter(args.sft_type) and args.ckpt_dir is not None:
        if isinstance(args, DeployArguments) and args.lora_request_list is not None:
            logger.info(f'args.lora_request_list: {args.lora_request_list}')
            for lora_request in args.lora_request_list:
                model = Swift.from_pretrained(
                    model, lora_request.lora_local_path, lora_request.lora_name, inference_mode=True)
        else:
            model = Swift.from_pretrained(model, args.ckpt_dir, inference_mode=True)
        model = model.to(model.dtype)
    model.requires_grad_(False)

    if task == 'infer':
        show_layers(model)
        logger.info(model)
    logger.info(get_model_info(model))
    template: Template = get_template(
        args.template_type,
        tokenizer,
        args.system,
        args.max_length,
        args.truncation_strategy,
        model=model,
        tools_prompt=args.tools_prompt)
    logger.info(f'system: {template.default_system}')
    return model, template

# read_media_file function: 读入媒体文件
    # infer_kwargs: 存储推理相关参数
    # infer_media_type: 媒体推理类型，可能的值有[none, round, dialogue, interleave]
    # media_type: 媒体类型，可能的值有[image, video, audio]
    # query: 用户输入字符串
def read_media_file(infer_kwargs: Dict[str, Any], infer_media_type: Literal['none', 'round', 'dialogue', 'interleave'],
                    media_type: Literal['image', 'video', 'audio'], query: str) -> None:
    # 如果infer_media_type为none，则直接返回
    if infer_media_type == 'none':
        return

    # 辅助函数_input_media，用于处理用户输入的媒体文件路径或url
    def _input_media(media_type: Literal['image', 'video', 'audio']) -> None:
        # media_type -> media_key:
            # image -> images
            # audio -> audios
            # video -> videos
        media_key = MediaTag.media_keys[media_type]
        # 从infer_kwargs中获取现有的媒体文件列表
        media_files = infer_kwargs.get(media_key) or []
        # a_an处理元音字母前缀
        a_an = 'an' if media_type[0] in {'i', 'a'} else 'a'
        # text用来提示输入media_type的path or url
        text = f'Input {a_an} {media_type} path or URL <<< '
        # 添加媒体文件至media_files数组
        media_files += [input(text) or None]
        # 重新存储至infer_kwargs中
        infer_kwargs[media_key] = media_files

    # 如果媒体推断类型为'interleave'
    if infer_media_type == 'interleave':
        # 使用正则表达式寻找匹配的标签
            # image -> <image>
            # audio -> <audio>
            # video -> <video>
        media_tags = re.findall('|'.join(list(MediaTag.standard_tags.values())), query)
        standard_tags_r = {v: k for k, v in MediaTag.standard_tags.items()}
        # 对每个匹配的标签，要求用户输入对应的多媒体文件
        for tag in media_tags:
            media_type = standard_tags_r[tag]
            _input_media(media_type)
        return

    # 如果是其余媒体推断类型
        # 先获取当前media_type对应的media_files
        # 如果推理类型为round或media_files为空，则提示输入文件
    media_key = MediaTag.media_keys[media_type]
    media_files = infer_kwargs.get(media_key) or []
    if infer_media_type == 'round' or len(media_files) == 0:
        _input_media(media_type)


def llm_infer(args: InferArguments) -> Dict[str, List[Dict[str, Any]]]:
    # 打印参数
    logger.info(f'args: {args}')
    # 设置种子
    seed_everything(args.seed)
    # 如果args中设置了merge_lora==true，则调用merge_lora进行lora合并
    if args.merge_lora:
        merge_lora(args, device_map=args.merge_device_map)
    # 根据不同的推理后端，导入相应模块和初始化模版
    ### infer_backend == 'vllm'
    if args.infer_backend == 'vllm':
        from .utils import (prepare_vllm_engine_template, inference_stream_vllm as inference_stream_x, inference_vllm as
                            inference_x)
        llm_engine, template = prepare_vllm_engine_template(args)
    ### infer_backend == 'lmdeploy'
    elif args.infer_backend == 'lmdeploy':
        from .utils import (prepare_lmdeploy_engine_template, inference_stream_lmdeploy as inference_stream_x,
                            inference_lmdeploy as inference_x)
        llm_engine, template = prepare_lmdeploy_engine_template(args)
    ### other infer_backend 
    else:
        # 量化配置
        if args.quant_method == 'hqq':
            from transformers import HqqConfig
            if args.hqq_dynamic_config_path is not None:
                cwd = os.getcwd()
                config_path = args.hqq_dynamic_config_path if os.path.isabs(
                    args.hqq_dynamic_config_path) else os.path.join(cwd, args.hqq_dynamic_config_path)
                with open(config_path, 'r') as json_file:
                    args.quant_config = HqqConfig(dynamic_config=json.load(json_file))
            else:
                if args.quantization_bit == 0:
                    logger.info("You haven't set the quantization_bit parameter; set it to 8.")
                    args.quantization_bit = 8
                args.quant_config = HqqConfig(nbits=args.quantization_bit, axis=args.hqq_axis)
        elif args.quant_method == 'eetq':
            from transformers import EetqConfig
            args.quant_config = EetqConfig('int8')
        # 根据args配置模型和模版
        model, template = prepare_model_template(args, device_map=args.device_map_config)
        # 如果设置了args.overwrite_generation_config
            # 需确保args.ckpt_dir不为空，将generation_config写入ckpt_dir
        if args.overwrite_generation_config:
            assert args.ckpt_dir is not None, 'args.ckpt_dir is not specified.'
            model.generation_config.save_pretrained(args.ckpt_dir)
    # lora请求初始化
        # 先将lora_request初始化为None
        # 如果args.vllm_enable_lora为True，则将lora_request_list[0]赋值给lora_request
    lora_request = None
    if args.vllm_enable_lora:
        assert len(args.lora_request_list) == 1
        lora_request = args.lora_request_list[0]
    # Inference
    # 结果存储
    result: List[Dict[str, Any]] = []
    jsonl_path = None
    if args.save_result:
        if args.result_dir:
            result_dir = args.result_dir
        else:
            result_dir = args.ckpt_dir
            if result_dir is None:
                result_dir = llm_engine.model_dir if args.infer_backend in {'vllm', 'lmdeploy'} else model.model_dir
            if result_dir is not None:
                result_dir = os.path.join(result_dir, 'infer_result')
        if result_dir is not None:
            os.makedirs(result_dir, exist_ok=True)
            time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
            # jsonl_path为result_dir下基于当前时间的jsonl文件
            jsonl_path = os.path.join(result_dir, f'{time}.jsonl')
    ### args.eval_human决定是否启用人机交互
    if args.eval_human:
        # input_mode: 
            # S为Single-line，表示单行输入模式
            # M为Multi-line，表示多行输入模式
        input_mode: Literal['S', 'M'] = 'S'
        logger.info('Input `exit` or `quit` to exit the conversation.')
        logger.info('Input `multi-line` to switch to multi-line input mode.')
        logger.info('Input `reset-system` to reset the system and clear the history.')
        # 检查template是否支持多轮对话
        if template.support_multi_round:
            logger.info('Input `clear` to clear the history.')
        else:
            logger.info('The current template only supports single-round dialogues.')
        # history用于存储历史
        history = []
        infer_kwargs = {}
        if args.infer_media_type != 'none':
            logger.info('Please enter the conversation content first, followed by the path to the multimedia file.')
        # system用于存储系统命令（如You are a helpful assistant)
        # read_system指示当前是否处于读取系统命令状态
        system = None
        read_system = False
        # 主循环处理用户输入
        while True:
            ### Single-line mode
            if input_mode == 'S':
                # addi_prompt用于在提示符中显示当前模式，方便用户识别
                # addi_prompt中[S]为system，[M]为multi-line
                addi_prompt = ''
                if read_system:
                    addi_prompt = '[S]'
                query = input(f'<<<{addi_prompt} ')
            ### Multi-line mode
            else:
                addi_prompt = '[M]'
                # 如果当前在等待系统命令，则将addi_prompt变为[MS]
                if read_system:
                    addi_prompt = '[MS]'
                # read_multi_line用于读取多行输入，直到满足特定条件，query为所有的多行输入拼接（不包括换行符）
                # 多行输入只是保证输入可以多行分开，增加可读性
                query = read_multi_line(addi_prompt)
            ### query为该轮所有输入的内容
            # query在{exit, quit}中，则退出系统
            if query.strip().lower() in {'exit', 'quit'}:
                break
            # query为clear，则重置history和infer_kwargs，continue继续下一轮输入
            elif query.strip().lower() == 'clear':
                history = []
                infer_kwargs = {}
                continue
            # 如果query为空字符串，且当前并不是read_system状态，则继续
            elif query.strip() == '' and not read_system:
                continue
            # 如果query为reset-system，将read_system置为True
            elif query.strip().lower() == 'reset-system':
                read_system = True
                continue
            # 如果此时read_system为True
                # 重置历史对话history
                # 更新system为当前输入的query（即系统命令）
            if read_system:
                if query == '':
                    system = None
                else:
                    system = query
                read_system = False
                history = []
                infer_kwargs = {}
                continue
            ### 以下代表read_system为False，即输入非系统指令
            # 如果当前input_mode为'S'且输入query为multi-line，则转变input_mode为M
            if input_mode == 'S' and query.strip().lower() == 'multi-line':
                input_mode = 'M'
                logger.info('End multi-line input with `#`.')
                logger.info('Input `single-line` to switch to single-line input mode.')
                continue
            # 如果当前input_mode为'M'且输入query为single-line，则转变input_mode为S
            if input_mode == 'M' and query.strip().lower() == 'single-line':
                input_mode = 'S'
                continue
            # 如果当前模板不支持多轮对话，重置history和infer_kwargs
            if not template.support_multi_round:
                history = []
                infer_kwargs = {}
            ### -------------------以上还未对对话内容的query作推理-------------------- ###
            # import pdb; pdb.set_trace()
            # 读入媒体文件（image、video、audio），包含在infer_kwargs[media_key]中
            read_media_file(infer_kwargs, args.infer_media_type, args.media_type, query)
            # 将args.truncation_strategy存储至infer_kwargs中
            infer_kwargs['truncation_strategy'] = args.truncation_strategy
            # 设置system
            if system is None and template.use_default_system:
                system = template.default_system
            ### 如果推理后端在{'vllm', 'lmdeploy'}中
            if args.infer_backend in {'vllm', 'lmdeploy'}:
                # 创建一个请求列表
                request_list = [{'query': query, 'history': history, 'system': system, **infer_kwargs}]
                # 如果使用流式推理，调用inference_stream_x进行流式推理
                if args.stream:
                    gen = inference_stream_x(llm_engine, template, request_list, lora_request=lora_request)
                    print_idx = 0
                    for resp_list in gen:
                        response = resp_list[0]['response']
                        new_history = resp_list[0]['history']
                        if len(response) > print_idx:
                            print(response[print_idx:], end='', flush=True)
                            print_idx = len(response)
                    print()
                # 不使用流式推理，调用inference_x进行非流式推理
                else:
                    resp_list = inference_x(llm_engine, template, request_list, lora_request=lora_request)
                    response = resp_list[0]['response']
                    new_history = resp_list[0]['history']
                    print(response)
            ### 推理后端非vllm和lmdeploy 
            else:
                # 检查是否有停用词
                if args.stop_words:
                    infer_kwargs['stop_words'] = args.stop_words
                # 如果使用流式推理，调用inference_stream函数进行流式推理
                if args.stream:
                    gen = inference_stream(model, template, query, history, system, **infer_kwargs)
                    print_idx = 0
                    for response, new_history in gen:
                        if len(response) > print_idx:
                            print(response[print_idx:], end='', flush=True)
                            print_idx = len(response)
                    print()
                # 非流式推理，则调用inference函数进行非流式推理
                else:
                    response, new_history = inference(model, template, query, history, system, **infer_kwargs)
                    print(response)
            # 打印分隔符
            print('-' * 50)
            # 构造结果对象
            obj = {
                'system': system,
                'query': query,
                'response': response,
                'history': history,
            }
            # 往结果对象obj中加入输入的媒体文件路径
            for media_key in MediaTag.media_keys.values():
                media_files = infer_kwargs.get(media_key)
                if media_files is not None:
                    obj[media_key] = media_files
            # 更新历史记录
            history = new_history
            # 将结果添加至jsonl_path中
            if jsonl_path is not None:
                append_to_jsonl(jsonl_path, obj)
            # 存储当前结果至result中
            result.append(obj)
    
    ### args.eval_ficl决定是否执行ficl指标评估
    elif args.eval_ficl:

        # ficl data
        ficl_example_data = [
            {
                "Example": 0,
                "History_title": ["孤独的人总被困在原地", "我真的抓住了风"],
                "History_imgs": ["Group_1-4N1A7C-1.png", "Group_1-4N1A7C-2.png"],
                "Target_img": "Group_1-4N1A7C-3.png",
                "Generated_title": "光影浮动，守住内心的热爱",
                "Ground_truth": "洗涤心灵的乡村初秋之旅",
                "Emotion_similarity": 1,
                "Language_Style_similarity": 0.75,
                "Content_Relevance_similarity": 0.75
            },
            {
                "Example": 1,
                "History_title": ["回村过夏天｜蓝天白云和稻田☁️", "荷风送清香｜荷花清新淡雅感调色教程"],
                "History_imgs": ["Group_1-2K3I8P-1.png", "Group_1-2K3I8P-2.png"],
                "Target_img": "Group_1-2K3I8P-3.png",
                "Generated_title": "田园夏日｜一抹夏日清新",
                "Ground_truth": "在绿色夏日里｜这个清透绿色调好适合夏天",
                "Emotion_similarity": 1.0,
                "Language_Style_similarity": 1.0,
                "Content_Relevance_similarity": 1.0
            },
            {
                "Example": 2,
                "History_title": ["回村过夏天｜蓝天白云和稻田☁️", "荷风送清香｜荷花清新淡雅感调色教程"],
                "History_imgs": ["Group_1-2K3I8P-1.png", "Group_1-2K3I8P-2.png"],
                "Target_img": "Group_1-2K3I8P-3.png",
                "Generated_title": "绿色是风的形状",
                "Ground_truth": "在绿色夏日里｜这个清透绿色调好适合夏天",
                "Emotion_similarity": 1.0,
                "Language_Style_similarity": 0.25,
                "Content_Relevance_similarity": 0.5
            },
            {
                "Example": 3,
                "History_title": ["三趾滨鹬路过深圳啦", "灰奇鹛到底有多奇？"],
                "History_imgs": ["Group_1-9Y3U5X-1.png", "Group_1-9Y3U5X-2.png"],
                "Target_img": "Group_1-9Y3U5X-3.png",
                "Generated_title": "秋天的风景",
                "Ground_truth": "来数数：尾巴有12根线的十二线极乐鸟",
                "Emotion_similarity": 0,
                "Language_Style_similarity": 0,
                "Content_Relevance_similarity": 0
            },
            {
                "Example": 4,
                "History_title": ["窗边的独居小卧室", "极简ins风的独居卧室"],
                "History_imgs": ["Group_1-9Z0Q4K-1.png", "Group_1-9Z0Q4K-2.png"],
                "Target_img": "Group_1-9Z0Q4K-3.png",
                "Generated_title": "山间小屋：冬日里的宁静角落",
                "Ground_truth": "有一扇被四季独宠的窗~",
                "Emotion_similarity": 0.5,
                "Language_Style_similarity": 0.25,
                "Content_Relevance_similarity": 0.5
            },
            {
                "Example": 5,
                "History_title": ["一些可爱的小细节", "日常生活中的浪漫瞬间"],
                "History_imgs": ["Group_1-4R9Y3J-1.png", "Group_1-4R9Y3J-2.png"],
                "Target_img": "Group_1-4R9Y3J-3.png",
                "Generated_title": "自然之美：手工艺里的诗意时光",
                "Ground_truth": "自然元素的橡皮章！",
                "Emotion_similarity": 0.5,
                "Language_Style_similarity": 0.5,
                "Content_Relevance_similarity": 0.75
            },
            {
                "Example": 6,
                "History_title": ["一些可爱的小细节", "日常生活中的浪漫瞬间"],
                "History_imgs": ["Group_1-4R9Y3J-1.png", "Group_1-4R9Y3J-2.png"],
                "Target_img": "Group_1-4R9Y3J-3.png",
                "Generated_title": "可爱的小仙人掌钩针图解",
                "Ground_truth": "自然元素的橡皮章！",
                "Emotion_similarity": 0.75,
                "Language_Style_similarity": 0.5,
                "Content_Relevance_similarity": 0.25
            },
            {
                "Example": 7,
                "History_title": ["上海周末City", "上海周末City"],
                "History_imgs": ["Group_1-3V9M0K-1.png", "Group_1-3V9M0K-2.png"],
                "Target_img": "Group_1-3V9M0K-3.png",
                "Generated_title": "可爱的小黄人",
                "Ground_truth": "上海周末City",
                "Emotion_similarity": 0,
                "Language_Style_similarity": 0,
                "Content_Relevance_similarity": 0
            },
            {
                "Example": 8,
                "History_title": ["上海周末City", "上海周末City"],
                "History_imgs": ["Group_1-3V9M0K-1.png", "Group_1-3V9M0K-2.png"],
                "Target_img": "Group_1-3V9M0K-3.png",
                "Generated_title": "图片展示了一片绿油油的树林和一条小路,穿过树林",
                "Ground_truth": "上海周末City",
                "Emotion_similarity": 0,
                "Language_Style_similarity": 0,
                "Content_Relevance_similarity": 0
            },
            {
                "Example": 9,
                "History_title": ["上海周末City", "上海周末City"],
                "History_imgs": ["Group_1-3V9M0K-1.png", "Group_1-3V9M0K-2.png"],
                "Target_img": "Group_1-3V9M0K-3.png",
                "Generated_title": "逃离都市喧嚣，感受自然宁静——上海周末city的绿色避风港",
                "Ground_truth": "上海周末City",
                "Emotion_similarity": 1,
                "Language_Style_similarity": 0.5,
                "Content_Relevance_similarity": 1
            }
        ]

        # check
        if args.ficl_example_num > len(ficl_example_data):
            raise ValueError(f'set ficl_example_num: {args.ficl_example_num} exceeds limitation ...')
        
        # construct public data
        img_dir = args.img_dir
        img_list = []
        base_prompt = "From 0 (not similar or relevant at all)，0.25(slightly similar), 0.5(similar in some aspects), 0.75(similar in many aspects) to 1(almost identical), rate the similarity between generated comment and ground-truth in terms of emotion, language style, and content relevance. Generate your final output in form of “Emotion: x, Style: y, Relevance: z”, where x, y and z are the scores you pick for each dimentsion. Only the score needs to be output, and there is no need to explain the reason."
        example_prompt = ""

        # fill in example_prompt
        for i in range(args.ficl_example_num):
            example_data = ficl_example_data[i]
            prompt = "Example {}: ".format(example_data["Example"])
            if args.use_ficl_history_img and args.use_ficl_history_title:
                history_title_list = example_data["History_title"]
                for j, history_title in enumerate(history_title_list):
                    prompt = prompt + f"{j+1} History title: {history_title}, corresponding to images: <image>."
                    img_list.append(example_data["History_imgs"][j])

            elif args.use_ficl_history_title:
                history_title_list = example_data["History_title"]
                prompt = prompt + f"History title: {example_data['History_title']}."
                
            elif args.use_ficl_history_img:
                prompt = prompt + f"History images: {'<image>' * len(example_data['History_imgs'])}."
                img_list = img_list + example_data['History_imgs']
            
            if args.use_ficl_target_img:
                prompt = prompt + "Target image: <image>"
                img_list.append(example_data['Target_img'])
            prompt = prompt + f"Generated title: {example_data['Generated_title']}, Ground truth: {example_data['Ground_truth']},"
            prompt = prompt + f"Emotion similarity: {example_data['Emotion_similarity']}, Language Style similarity: {example_data['Language_Style_similarity']}, Content Relevance similarity: {example_data['Content_Relevance_similarity']};"
        
            example_prompt = example_prompt + prompt


        # run item
        if args.ficl_sample_dataset is None or args.ficl_model_gen_path is None or args.ficl_save_dir is None:
            
            data = {'system': 'you are an intelligent auto-rater for personalized title generation task.'}
            target_prompt = "Now let's get started ! Generated title: {}, Ground truth: {}" 
            # fill in target prompt 
            target_prompt = target_prompt.format('蓝天下的温柔诗意｜静谧悠然的花语时光', '九月｜格桑花和秋天一起悄悄地来了') # 测试数据
            ficl_prompt = base_prompt + example_prompt + target_prompt

            # fill in data <images>
            for i, img_prefix in enumerate(img_list):
                img_list[i] = os.path.join(img_dir, img_prefix)
            if len(img_list) > 0:
                data['images'] = img_list

            # 构造推理kwargs
            kwargs = {'query': ficl_prompt}
    
            # 获取data参数
            history = data.get('history')
            system = data.get('system')
            # import pdb; pdb.set_trace()
            tools = data.get('tools')
            objects = data.get('objects')
            # 如果采用了详尽输出且system不为None，打印system
            if args.verbose and system is not None:
                print(f'[SYSTEM]{system}')
            # ------处理历史对话------
            if history is None:
                history = []
            # 加入kwargs字典
            kwargs['history'] = history
            # ------处理system------
            if system is None and template.use_default_system:
                system = template.default_system
            # 加入kwargs字典
            kwargs['system'] = system
            # ------处理多媒体文件------
            # 遍历media_key: [images, audios, videos]
            for media_key in MediaTag.media_keys.values():
                media_files = data.get(media_key)
                # 加入kwargs字典
                if media_files is not None:
                    kwargs[media_key] = media_files
            # ------处理tools、objects、truncation_strategy------
            if tools is not None:
                kwargs['tools'] = tools
            if objects is not None:
                kwargs['objects'] = objects
            kwargs['truncation_strategy'] = args.truncation_strategy
            if args.infer_backend in {'vllm', 'lmdeploy'}:
                assert args.stream
                if args.verbose:
                    print(f"[QUERY]{data['query']}\n[RESPONSE]", end='')
                gen = inference_stream_x(llm_engine, template, [kwargs], lora_request=lora_request)
                print_idx = 0
                for resp_list in gen:
                    response = resp_list[0]['response']
                    if args.verbose and len(response) > print_idx:
                        print(response[print_idx:], end='', flush=True)
                        print_idx = len(response)
                print()
            # 调用inference接口得到response
            else:
                response, _ = inference(
                    model, template, stream=args.stream and args.verbose, verbose=args.verbose, **kwargs)
            # 若data中包含了标签信息，赋值为label
            label = data.pop('response', None)
            # 构建对话保存实例
            obj = {
                'system': kwargs['system'],
                'query': kwargs['query'],
                'response': response,
                'label': label,
                'history': kwargs['history'],
            }
            # 将多媒体文件路径加入对话保存实例
            for media_key in MediaTag.media_keys.values():
                media_files = kwargs.get(media_key)
                if media_files is not None:
                    obj[media_key] = media_files
            # 输出到对话存储文件中
            if jsonl_path is not None:
                append_to_jsonl(jsonl_path, obj)
            # 将对话保存实例加入result中
            result.append(obj)
            if args.verbose:
                print()
                print(f'[LABELS]{label}')
                for media_key in MediaTag.media_keys.values():
                    media_files = kwargs.get(media_key)
                    if media_files is not None:
                        print(f'[{media_key.upper()}]{media_files}')
                print('-' * 50, flush=True)

        # run all sample-data
        else:
            # iter
            sample_data = load_json(args.ficl_sample_dataset)
            model_gen_data = load_json(args.ficl_model_gen_path)
            # result
            ficl_result = {} 
            for group_name, group_value in sample_data.items():
                anonymous_list = [name for name in group_value if name != 'count']
                for anonymous_name in tqdm(anonymous_list):
                    ficl_result[anonymous_name] = {}
                    anonymous_data = group_value[anonymous_name]
                    ground_title = anonymous_data['target']['title']
                    target_base_prompt = "Now let's get started ! "
                   
                    for model_name, model_gen in model_gen_data.items():
                       
                        gen_title = model_gen[anonymous_name]
                        target_necessary_prompt = "Generated title: {}, Ground truth: {}".format(gen_title, ground_title)
                        ficl_img_list = img_list.copy()
                        target_control_prompt = ""
                        if args.use_ficl_history_title and args.use_ficl_history_img:
                            for i, history_note in enumerate(anonymous_data['history']):
                                target_control_prompt = target_control_prompt + f"{i+1} History title: {history_note['title']}, corresponding to images: <image>."
                                ficl_img_list.append(history_note['sample_prefix'])

                        elif args.use_ficl_history_title:
                            history_title_list = [history_note['title'] for history_note in anonymous_data['history']]
                            target_control_prompt = target_control_prompt + f"History titles: {history_title_list}."
                        
                        elif args.use_ficl_history_img:
                            history_img_list = [history_note['sample_prefix'] for history_note in anonymous_data['history']]
                            target_control_prompt = target_control_prompt + f"History images: {'<image>' * len(anonymous_data['history'])}."
                            ficl_img_list = ficl_img_list + history_img_list

                        else:
                            target_control_prompt = ""
                        
                        if args.use_ficl_target_img:
                            target_control_prompt = target_control_prompt + "Target image: <image>."
                            ficl_img_list = ficl_img_list + [anonymous_data['target']['sample_prefix']]

                        ficl_img_list = [os.path.join(img_dir, img_prefix) for img_prefix in ficl_img_list]
                        ficl_target_prompt = target_base_prompt + target_control_prompt + target_necessary_prompt
                        ficl_prompt = base_prompt + example_prompt + ficl_target_prompt

                        data = {'system': 'you are an intelligent auto-rater for personalized title generation task.'}
                        if len(ficl_img_list) > 0:
                            data['images'] = ficl_img_list

                        # 构造推理kwargs
                        kwargs = {'query': ficl_prompt}
                
                        # 获取data参数
                        history = data.get('history')
                        system = data.get('system')
                        # import pdb; pdb.set_trace()
                        tools = data.get('tools')
                        objects = data.get('objects')
                        # 如果采用了详尽输出且system不为None，打印system
                        if args.verbose and system is not None:
                            print(f'[SYSTEM]{system}')
                        # ------处理历史对话------
                        if history is None:
                            history = []
                        # 加入kwargs字典
                        kwargs['history'] = history
                        # ------处理system------
                        if system is None and template.use_default_system:
                            system = template.default_system
                        # 加入kwargs字典
                        kwargs['system'] = system
                        # ------处理多媒体文件------
                        # 遍历media_key: [images, audios, videos]
                        for media_key in MediaTag.media_keys.values():
                            media_files = data.get(media_key)
                            # 加入kwargs字典
                            if media_files is not None:
                                kwargs[media_key] = media_files
                        # ------处理tools、objects、truncation_strategy------
                        if tools is not None:
                            kwargs['tools'] = tools
                        if objects is not None:
                            kwargs['objects'] = objects
                        kwargs['truncation_strategy'] = args.truncation_strategy
                        if args.infer_backend in {'vllm', 'lmdeploy'}:
                            assert args.stream
                            if args.verbose:
                                print(f"[QUERY]{data['query']}\n[RESPONSE]", end='')
                            gen = inference_stream_x(llm_engine, template, [kwargs], lora_request=lora_request)
                            print_idx = 0
                            for resp_list in gen:
                                response = resp_list[0]['response']
                                if args.verbose and len(response) > print_idx:
                                    print(response[print_idx:], end='', flush=True)
                                    print_idx = len(response)
                            print()
                        # 调用inference接口得到response
                        else:
                            response, _ = inference(
                                model, template, stream=args.stream and args.verbose, verbose=args.verbose, **kwargs)
                        # 若data中包含了标签信息，赋值为label
                        label = data.pop('response', None)
                        # 构建对话保存实例
                        obj = {
                            'system': kwargs['system'],
                            'query': kwargs['query'],
                            'response': response,
                            'label': label,
                            'history': kwargs['history'],
                        }
                        # 将多媒体文件路径加入对话保存实例
                        for media_key in MediaTag.media_keys.values():
                            media_files = kwargs.get(media_key)
                            if media_files is not None:
                                obj[media_key] = media_files
                        # 输出到对话存储文件中
                        if jsonl_path is not None:
                            append_to_jsonl(jsonl_path, obj)

                        # 加入ficl_result
                        ficl_result[anonymous_name][model_name] = response

                        # 将对话保存实例加入result中
                        result.append(obj)
                        if args.verbose:
                            print()
                            print(f'[LABELS]{label}')
                            for media_key in MediaTag.media_keys.values():
                                media_files = kwargs.get(media_key)
                                if media_files is not None:
                                    print(f'[{media_key.upper()}]{media_files}')
                            print('-' * 50, flush=True)

            unique_name = f'{args.model_type}-{args.ficl_example_num}-{int(args.use_ficl_history_title)}-{int(args.use_ficl_history_img)}-{int(args.use_ficl_target_img)}'
            unique_index = 0
            while os.path.exists(os.path.join(args.ficl_save_dir, f'{unique_name}#{unique_index}.json')):
                unique_index += 1
            ficl_save_path = os.path.join(args.ficl_save_dir, f'{unique_name}#{unique_index}.json')
            with open(ficl_save_path, 'w', encoding='utf-8') as f:
                json.dump(ficl_result, f, ensure_ascii=False, indent=4)
            print('-----------------ficl推理完成-------------')

    ### args.eval_udcf决定是否执行udcf指标评估 (后更新为FME-Score)
    elif args.eval_udcf:

        # udcf args
        udcf_rank_num = args.udcf_rank_num
        udcf_example_num = args.udcf_example_num
        use_udcf_history_img = args.use_udcf_history_img
        udcf_streamline = args.enable_udcf_streamline
        eval_mode = "NDCG"
        # data
        sample_data = load_json(args.udcf_sample_dataset)
        group_data = load_json(args.group_index_path)
        img_dir = args.img_dir
        # 若aba_model_name和aba_model_gen_path均不为空，则执行的是消融评估数据
        if args.aba_model_name is not None and args.aba_model_gen_path is not None:
            gen_data = load_json(args.aba_model_gen_path)[args.aba_model_name]
            eval_mode = "Aba"
        else:
            gen_data = load_json(args.udcf_model_gen_path)

        # udcf example data
        udcf_examples = [
            {
                "anonymous_name": "2O4P7O",
                "portrait_analysis": {
                    "Personal_Biography": "The user describes themselves as someone drawn to 'everything in this world', demonstrating a strong exploratory desire toward natural landscapes.",
                    "Historical Posts": "The user's historical post titles frequently mention geographic locations and descriptions of natural beauty, such as 'Qingdao', 'Lianyungang' and 'Ancient Architecture in Shanxi'. This suggests a preference for highlighting unique features of Chinese landscapes.",
                    "Latest Post": "The user's latest post's ground-truth title uses the word 'shocking' to describe a Buddha statue, reflecting his habit of expressing emotional shifts in titles rather than relying on objective descriptions alone."
                },
                "model_evaluation_process": {
                    "criteria": "Please follow these steps to evaluate and rank captions generated by models (e.g., model_A to model_H) for the user's latest post:",
                    "Screening and Exclusion": "Identify and exclude models with clearly low-quality captions that fail to align with the user's profile and habits (e.g. model_A、model_C、model_D).",
                    "Repetition Check": "Penalize models that merely repeat past titles (e.g., model_B), classifying them as low quality. For models like model_F and model_G that exhibit expansive repetition—building on historical themes but with limited creativity, apply minor penalties in the ranking."
                },
                "emotional_and_expression_habit_analysis": {
                    "Expression_Habits": "Captions aligning with the user's tendency to mention geographic locations in their titles should be rated higher.",
                    "Emotion State": "Prioritize captions that convey the user's exploratory spirit and emotional shifts, such as 'shocking' or 'amazing'."
                },
                "short_reason": "Identify and exclude models with clearly low-quality captions(e.g. model_A、model_C、model_D). Penalize models that merely repeat past titles (e.g., model_B), classifying them as low quality. Prioritize captions that convey the user's exploratory spirit and emotional shifts.",
                "ground_output": "{{Rankings: model_H > model_F > model_E > model_G, Reasons: Identify and exclude models with clearly low-quality captions(e.g. model_A、model_C、model_D). Penalize models that merely repeat past titles (e.g., model_B), classifying them as low quality. Prioritize captions that convey the user's exploratory spirit and emotional shifts.}}"
            },
            {
                "anonymous_name": "2K3I8P",
                "portrait_analysis": {
                    "Personal_Biography": "The user's personal biography demonstrates that she is a photography artist striving for excellence, with a passion for capturing the beauty of life.",
                    "Historical Posts": "The user's historical titles follow a specific format: xxx｜yyy, and reflect her attention to color grading.",
                    "Latest Post": "The user's latest post still focuses on the colors displayed in the image and adheres to her inherent format."
                },
                "model_evaluation_process": {
                    "criteria": "Please follow these steps to evaluate and rank captions generated by models (e.g., model_A to model_H) for the user's latest post:",
                    "Screening and Exclusion": "Identify and exclude models with clearly low-quality captions that fail to align with the user's profile and habits (e.g. model_A、model_D、model_E).",
                    "Repetition Check": "Penalize models that merely repeat past titles (e.g., model_B), classifying them as low quality."
                },
                "emotional_and_expression_habit_analysis": {
                    "Expression_Habits": "Captions aligning with the user's inherent text format(xxx｜yyy) should be rated higher.",
                    "Emotion State": "Captions that avoid simple repetition, follow the user's text format, and depict colors should be preferred over those that merely describe the image content."
                },
                "short_reason": "Identify and exclude models with clearly low-quality captions(e.g. model_A、model_D、model_E). Penalize models that merely repeat past titles (e.g., model_B), classifying them as low quality. Captions that avoid simple repetition, follow the user's text format, and depict colors should be preferred over those that merely describe the image content.",
                "ground_output": "{{Rankings: model_H > model_C > model_G = model_F, Reasons: Identify and exclude models with clearly low-quality captions(e.g. model_A、model_D、model_E). Penalize models that merely repeat past titles (e.g., model_B), classifying them as low quality. Captions that avoid simple repetition, follow the user's text format, and depict colors should be preferred over those that merely describe the image content.}}"
            }
        ]

        # base prompt
        udcf_base_prompt = "You are tasked with evaluating the quality of personalized image captions generated by different models (e.g., model_A, model_B, ..., model_F). Each caption is created based on the user's profile, post history (including past images and captions), and a new image input. Your objective is to assess the captions' quality and rank them based on their alignment with the user's unique expression style, personal interests, and mood. "
        # navigation prompt
        udcf_navi_prompt = "To assist you in understanding the evaluation criteria, I will provide several examples of captions and their quality assessments. Use these examples as a reference to guide your judgment. "
        # output prompt 
        udcf_output_prompt = "Please provide your output in the format: {{Rankings: model_X > model_Y = model_Z > model_W, Reasons: xxx}}, where '>' indicates a higher quality caption and '=' indicates equally high quality. Rank only the top {} models based on the generated captions and briefly explain your reasoning under 'Reasons'. ".format(udcf_rank_num)

        # parse udcf-example
        def parse_udcf_example(udcf_example_num, udcf_base_prompt, streamline=True):
            # img store
            img_list = []

            # check udcf example num
            assert udcf_example_num <= len(udcf_examples), f'set udcf_example_num: {udcf_example_num} does not meet requirement.'

            if udcf_example_num == 0:
                udcf_prompt = udcf_base_prompt + udcf_output_prompt
                return udcf_prompt, img_list
            
            udcf_prompt = udcf_base_prompt + udcf_navi_prompt
            for i in range(udcf_example_num):
                # extract data
                udcf_example = udcf_examples[i]
                anonymous_name = udcf_example['anonymous_name']
                user_group = group_data[anonymous_name]
                anonymous_data = sample_data[user_group][anonymous_name]
                user = anonymous_data['user']
                history_list = anonymous_data['history']
                target_note = anonymous_data['target']
                # profile prompt
                user_profile_prompt = "[User Profile: sex: {}, job: {}, description: {}]. ".format(user['sex'], user['job'], user['desc_info'])
                # example_prompt
                udcf_example_prompt = f"Example {i+1}: " + user_profile_prompt
                if use_udcf_history_img:
                    user_history_prompt = "["
                    suffix_map = {1: '1st', 2:'2nd', 3:'3rd'}
                    for j, history_note in enumerate(history_list):
                        img_prefix = history_note['sample_prefix']
                        user_history_prompt = user_history_prompt + "{} History: image: <image>, title:{}. ".format(suffix_map[j+1], history_note['title'])
                        img_list.append(img_prefix)
                    user_history_prompt = user_history_prompt + "]. "
                else:
                    history_title_list = [history_note['title'] for history_note in history_list]
                    user_history_prompt = "[User History titles: {}]. ".format(history_title_list)
                # target prompt
                user_target_prompt = "New Post images: <image>. "
                img_list.append(target_note['sample_prefix'])
                # gen prompt
                base_letter_idx = ord('A')
                diff_gen_prompt = "[Different Models Generation: "
                for model_name, model_value in gen_data.items():
                    current_letter = chr(base_letter_idx)
                    gen_title = model_value[anonymous_name]
                    diff_gen_prompt = diff_gen_prompt + f"model_{current_letter}: {gen_title}. "
                    base_letter_idx += 1
                diff_gen_prompt = diff_gen_prompt + "]. "
                # reason prompt
                if streamline:
                    user_reason_prompt = "[System Output: {}]. ".format(udcf_example['ground_output'])
                else:
                    user_reason_prompt = ""
                    pass

                    
                udcf_example_prompt = udcf_example_prompt + user_history_prompt + user_target_prompt + diff_gen_prompt + user_reason_prompt
                udcf_prompt = udcf_prompt + udcf_example_prompt

            udcf_prompt = udcf_prompt + udcf_output_prompt
            return udcf_prompt, img_list


        # parse target-anonymous
        def parse_target_anonymous(anonymous_name, udcf_prompt, img_list):
            target_prompt = "Now let's get started! " 
            # extract data
            user_group = group_data[anonymous_name]
            anonymous_data = sample_data[user_group][anonymous_name]
            user = anonymous_data['user']
            history_list = anonymous_data['history']
            target_note = anonymous_data['target']
            # profile prompt
            user_profile_prompt = "[User Profile: sex: {}, job: {}, description: {}]. ".format(user['sex'], user['job'], user['desc_info'])
            if use_udcf_history_img:
                user_history_prompt = "["
                suffix_map = {1: '1st', 2:'2nd', 3:'3rd'}
                for j, history_note in enumerate(history_list):
                    img_prefix = history_note['sample_prefix']
                    user_history_prompt = user_history_prompt + "{} History: image: <image>, title:{}. ".format(suffix_map[j+1], history_note['title'])
                    img_list.append(img_prefix)
                user_history_prompt = user_history_prompt + "]. "
            else:
                history_title_list = [history_note['title'] for history_note in history_list]
                user_history_prompt = "[User History titles: {}]. ".format(history_title_list)
            # target prompt
            user_target_prompt = "New Post images: <image>. "
            img_list.append(target_note['sample_prefix'])
            # gen prompt
            base_letter_idx = ord('A')
            diff_gen_prompt = "[Different Models Generation: "
            for model_name, model_value in gen_data.items():
                current_letter = chr(base_letter_idx)
                gen_title = model_value[anonymous_name]
                diff_gen_prompt = diff_gen_prompt + f"model_{current_letter}: {gen_title}. "
                base_letter_idx += 1
            diff_gen_prompt = diff_gen_prompt + "]. "
            query = udcf_prompt + target_prompt + user_profile_prompt + user_history_prompt + user_target_prompt + diff_gen_prompt
            query = query + udcf_output_prompt

            return query, img_list


        # run item
        if args.udcf_mode == 'test':
            data = {'system': 'you are an intelligent auto-rater for personalized title generation task.'}
            # add udcf example prompt
            udcf_prompt, img_list = parse_udcf_example(udcf_example_num=udcf_example_num, udcf_base_prompt=udcf_base_prompt, streamline=udcf_streamline)
            
            # For target anonymous_name
            query, img_list = parse_target_anonymous(anonymous_name='2D9P3R', udcf_prompt=udcf_prompt, img_list=img_list.copy())

            # fill in data <images>
            for i, img_prefix in enumerate(img_list):
                img_list[i] = os.path.join(img_dir, img_prefix)
            if len(img_list) > 0:
                data['images'] = img_list

            # 构造推理kwargs
            kwargs = {'query': query}
    
            # 获取data参数
            history = data.get('history')
            system = data.get('system')
            # import pdb; pdb.set_trace()
            tools = data.get('tools')
            objects = data.get('objects')
            # 如果采用了详尽输出且system不为None，打印system
            if args.verbose and system is not None:
                print(f'[SYSTEM]{system}')
            # ------处理历史对话------
            if history is None:
                history = []
            # 加入kwargs字典
            kwargs['history'] = history
            # ------处理system------
            if system is None and template.use_default_system:
                system = template.default_system
            # 加入kwargs字典
            kwargs['system'] = system
            # ------处理多媒体文件------
            # 遍历media_key: [images, audios, videos]
            for media_key in MediaTag.media_keys.values():
                media_files = data.get(media_key)
                # 加入kwargs字典
                if media_files is not None:
                    kwargs[media_key] = media_files
            # ------处理tools、objects、truncation_strategy------
            if tools is not None:
                kwargs['tools'] = tools
            if objects is not None:
                kwargs['objects'] = objects
            kwargs['truncation_strategy'] = args.truncation_strategy
            if args.infer_backend in {'vllm', 'lmdeploy'}:
                assert args.stream
                if args.verbose:
                    print(f"[QUERY]{data['query']}\n[RESPONSE]", end='')
                gen = inference_stream_x(llm_engine, template, [kwargs], lora_request=lora_request)
                print_idx = 0
                for resp_list in gen:
                    response = resp_list[0]['response']
                    if args.verbose and len(response) > print_idx:
                        print(response[print_idx:], end='', flush=True)
                        print_idx = len(response)
                print()
            # 调用inference接口得到response
            else:
                response, _ = inference(
                    model, template, stream=args.stream and args.verbose, verbose=args.verbose, **kwargs)
            # 若data中包含了标签信息，赋值为label
            label = data.pop('response', None)
            # 构建对话保存实例
            obj = {
                'system': kwargs['system'],
                'query': kwargs['query'],
                'response': response,
                'label': label,
                'history': kwargs['history'],
            }
            # 将多媒体文件路径加入对话保存实例
            for media_key in MediaTag.media_keys.values():
                media_files = kwargs.get(media_key)
                if media_files is not None:
                    obj[media_key] = media_files
            # 输出到对话存储文件中
            if jsonl_path is not None:
                append_to_jsonl(jsonl_path, obj)
            # 将对话保存实例加入result中
            result.append(obj)
            if args.verbose:
                print()
                print(f'[LABELS]{label}')
                for media_key in MediaTag.media_keys.values():
                    media_files = kwargs.get(media_key)
                    if media_files is not None:
                        print(f'[{media_key.upper()}]{media_files}')
                print('-' * 50, flush=True)

        # run dataset
        elif args.udcf_mode == 'dataset':
            # udcf result

            udcf_result = {}

            data = {'system': 'you are an intelligent auto-rater for personalized title generation task.'}
            # add udcf example prompt
            udcf_prompt, img_list = parse_udcf_example(udcf_example_num=udcf_example_num, udcf_base_prompt=udcf_base_prompt, streamline=udcf_streamline)
            
            # iter
            for group_name, group_value in tqdm(sample_data.items()):
                group_anonymous_list = [name for name in group_value if name != 'count']
                for anonymous_name in group_anonymous_list:
                    query, query_img_list = parse_target_anonymous(anonymous_name=anonymous_name, udcf_prompt=udcf_prompt, img_list=img_list.copy())
                    # import pdb; pdb.set_trace()
                    # fill in data <images>
                    for i, img_prefix in enumerate(query_img_list):
                        query_img_list[i] = os.path.join(img_dir, img_prefix)
                    if len(query_img_list) > 0:
                        data['images'] = query_img_list

                    # 构造推理kwargs
                    kwargs = {'query': query}
            
                    # 获取data参数
                    history = data.get('history')
                    system = data.get('system')
                    # import pdb; pdb.set_trace()
                    tools = data.get('tools')
                    objects = data.get('objects')
                    # 如果采用了详尽输出且system不为None，打印system
                    if args.verbose and system is not None:
                        print(f'[SYSTEM]{system}')
                    # ------处理历史对话------
                    if history is None:
                        history = []
                    # 加入kwargs字典
                    kwargs['history'] = history
                    # ------处理system------
                    if system is None and template.use_default_system:
                        system = template.default_system
                    # 加入kwargs字典
                    kwargs['system'] = system
                    # ------处理多媒体文件------
                    # 遍历media_key: [images, audios, videos]
                    for media_key in MediaTag.media_keys.values():
                        media_files = data.get(media_key)
                        # 加入kwargs字典
                        if media_files is not None:
                            kwargs[media_key] = media_files
                    # ------处理tools、objects、truncation_strategy------
                    if tools is not None:
                        kwargs['tools'] = tools
                    if objects is not None:
                        kwargs['objects'] = objects
                    kwargs['truncation_strategy'] = args.truncation_strategy
                    if args.infer_backend in {'vllm', 'lmdeploy'}:
                        assert args.stream
                        if args.verbose:
                            print(f"[QUERY]{data['query']}\n[RESPONSE]", end='')
                        gen = inference_stream_x(llm_engine, template, [kwargs], lora_request=lora_request)
                        print_idx = 0
                        for resp_list in gen:
                            response = resp_list[0]['response']
                            if args.verbose and len(response) > print_idx:
                                print(response[print_idx:], end='', flush=True)
                                print_idx = len(response)
                        print()
                    # 调用inference接口得到response
                    else:
                        response, _ = inference(
                            model, template, stream=args.stream and args.verbose, verbose=args.verbose, **kwargs)
                    # 若data中包含了标签信息，赋值为label
                    label = data.pop('response', None)
                    # 构建对话保存实例
                    obj = {
                        'system': kwargs['system'],
                        'query': kwargs['query'],
                        'response': response,
                        'label': label,
                        'history': kwargs['history'],
                    }
                    # 将多媒体文件路径加入对话保存实例
                    for media_key in MediaTag.media_keys.values():
                        media_files = kwargs.get(media_key)
                        if media_files is not None:
                            obj[media_key] = media_files
                    # 输出到对话存储文件中
                    if jsonl_path is not None:
                        append_to_jsonl(jsonl_path, obj)
                    # 将对话保存实例加入result中
                    result.append(obj)

                    # 将回复加入udcf_result
                    udcf_result[anonymous_name] = response

                    if args.verbose:
                        print()
                        print(f'[LABELS]{label}')
                        for media_key in MediaTag.media_keys.values():
                            media_files = kwargs.get(media_key)
                            if media_files is not None:
                                print(f'[{media_key.upper()}]{media_files}')
                        print('-' * 50, flush=True)

            # save
            if eval_mode == "NDCG":
                unique_name = f"{args.model_type}-{args.udcf_example_num}-{int(args.use_udcf_history_img)}-{int(args.enable_udcf_streamline)}"
                unique_index = 0
                while(os.path.exists(os.path.join(args.udcf_save_dir, f"{unique_name}#{unique_index}.json"))):
                    unique_index += 1
            elif eval_mode == "Aba":
                unique_name = f"Aba_{args.aba_model_name}-{args.udcf_example_num}-{int(args.use_udcf_history_img)}"
                unique_index = 0
                while(os.path.exists(os.path.join(args.udcf_save_dir, f"{unique_name}#{unique_index}.json"))):
                    unique_index += 1
            udcf_save_path = os.path.join(args.udcf_save_dir, f"{unique_name}#{unique_index}.json")
            with open(udcf_save_path, 'w', encoding='utf-8') as f:
                json.dump(udcf_result, f, ensure_ascii=False, indent=4)
            
            print('----------------UDCF评估完成-----------------')

        else:
            raise ValueError(f'set udcf_mode: {args.udcf_mode} wrong.')

    elif args.enforce_cot:
        '''
        执行cot指令
        '''

        # load cot dataset
        if os.path.isfile(args.cot_sample_dataset):
            cot_dataset = load_json(args.cot_sample_dataset)
        else:
            raise ValueError(f"Your provided cot_dataset: {args.cot_sample_dataset} doesn't meet requirement. ")
        
        # load gen file
        if args.inherient_gen_file is not None:
            inherient_gen_data = load_json(args.inherient_gen_file)
        else:
            inherient_gen_data = None

        group_data = load_json(args.group_index_path)
        img_dir = args.img_dir

        # cot args
        cot_streamline = args.cot_streamline
        use_cot_history_img = args.use_cot_history_img
        control_memory_bank = args.control_memory_bank

        # 执行可控制记忆块的版本
        if control_memory_bank:
            # Personalized CoTs
            Personalized_CoTs = {
                # visible personalized CoT
                "Step_0": {
                    "prompt": "你将扮演多媒体平台智能标题生成助手，根据博主的个人简介、历史发帖（包括发帖图像及对应标题），为其生成个性化的标题。请严格按照以下3个步骤进行思考和生成。",
                    "accept": [],
                    "keep_memory": True,
                    "inherient": False
                },
                "Step_1": {
                    "prompt": "提取个性和情绪信息：从博主的个人简介中提取关键的个性和情绪特征，包括职业、兴趣爱好、表达倾向以及整体情绪风格，请用简洁明了的语言总结这些信息。 输出格式为：兴趣爱好：XX，表达倾向：XXX，整体情绪风格：XXX。",
                    "accept": [],
                    "keep_memory": True,
                    "inherient": False,
                },
                "Step_2": {
                    "prompt": "分析文字和符号使用习惯：分析该博主历史发帖标题中的文字和符号表达习惯，具体分析以下几点：1. 是否存在固定的标题结构（如主副标题分隔、并列内容等）。2. 是否偏好使用特定符号（如“|”或“/”等），以及这些符号的功能和使用场景。3. 标题的总体风格：文学化、描述性还是现代化表达。如果历史标题使用了分隔符，请准确识别分隔符的类型、用法，并在生成标题时严格遵循相同的格式。",
                    "accept": [],
                    "keep_memory": True,
                    "inherient": False
                },
                "Step_3": {
                    "prompt": "生成个性化标题：根据上述分析结果（个人简介、历史发帖习惯），我将提供该博主的最新发帖图像，请你为该最新发帖图像生成严格遵循博主分隔符偏好的个性化标题。",
                    "accept": [],
                    "keep_memory": True,
                    "inherient": False,
                },
                # extract
                "Step_4":{
                    "prompt": "请提取以下这段文本中的标题。",
                    "accept": [],
                    "keep_memory": False,
                    "inherient": False
                },

                # 后续优化
                "Step_5": {
                    "prompt": "",
                    "accept": [],
                    "keep_memory": False,
                    "inherient": True
                }
            }
            # visible parse function
            def parse_visible_profile(anonymous_name: str, streamline=False):
                '''
                profile
                '''
                group_index = group_data[anonymous_name]
                anonymous_data = cot_dataset[group_index][anonymous_name]
                user = anonymous_data['user']
                profile_input = "[个人简介：性别：{}; 职业: {}; 自我描述：{}]；".format(user['sex'], user['job'], user['desc_info'])

                return (profile_input, [])

            def parse_visible_history(anonymous_name: str, use_history_img=False, streamline=False):
                '''
                history
                '''
                group_index = group_data[anonymous_name]
                anonymous_data = cot_dataset[group_index][anonymous_name]
                history_list = anonymous_data['history']
                img_list = []
                history_input = "历史发帖内容："
                # 使用原始图像
                if not streamline:
                    for i, history_note in enumerate(history_list):
                        history_input = history_input + f"{i+1}. "
                        if use_history_img:
                            history_input = history_input + f"标题：{history_note['title']}，对应图像：{len(history_note['image_urls']) * '<image>'}"
                        else:
                            history_input = history_input + f"标题：{history_note['title']}"
                        if i + 1 == len(history_list):
                            history_input = history_input + "。 "
                        else:
                            history_input = history_input + "; "
                        if use_history_img:
                            img_list = img_list + history_note['image_urls']
                # 使用拼接图像
                else:
                    for i, history_note in enumerate(history_list):
                        history_input = history_input + f"{i+1}. "
                        if use_history_img:
                            history_input = history_input + f"标题：{history_note['title']}，对应图像：<image>"
                        else:
                            history_input = history_input + f"标题：{history_note['title']}"
                        if i + 1 == len(history_list):
                            history_input = history_input + "。 "
                        else:
                            history_input = history_input + "; "
                        if use_history_img:
                            img_list.append(os.path.join(img_dir, history_note['sample_prefix']))
                history_input = "[" + history_input + "]"
                return (history_input, img_list)

            def parse_visible_target(anonymous_name: str, history_img_list: list, streamline=False):
                group_index = group_data[anonymous_name]
                anonymous_data = cot_dataset[group_index][anonymous_name]
                target_note = anonymous_data['target']
                # 使用原始图像
                if not streamline:
                    img_list = target_note['image_urls']
                    target_input = f"该博主的最新发帖图像为：{len(img_list) * '<image>'}，请执行生成个性化标题任务！直接输出标题即可。你生成的个性化标题为："
                    img_list = history_img_list + img_list
                # 使用拼接图像
                else:
                    img_list = [os.path.join(img_dir, target_note['sample_prefix'])]
                    img_list = history_img_list + img_list
                    target_input = f"该博主的最新发帖图像为：<image>，请执行生成个性化标题任务！直接输出标题即可。你生成的个性化标题为："
                return (target_input, img_list)

            def parse_visible_anonymous_data(anonymous_name, use_history_img=False, streamline=False):
                
                parse_res = {}
                # profile
                profile_input, _ = parse_visible_profile(anonymous_name=anonymous_name, streamline=streamline)
                parse_res['profile'] = {'input': profile_input, 'images': []}
                # history
                history_input, history_img_list = parse_visible_history(anonymous_name=anonymous_name, use_history_img=use_history_img, streamline=streamline)
                parse_res['history'] = {'input': history_input, 'images': history_img_list}
                # target
                target_input, target_img_list = parse_visible_target(anonymous_name=anonymous_name, history_img_list=history_img_list, streamline=streamline)
                parse_res['target'] = {'input': target_input, 'images': target_img_list}

                return parse_res

            # 记录中间每个步骤生成的回复
            cot_process = {}

            # run item
            if args.cot_mode == 'test':
                anonymous_list = ["7C8L7E", "4N1A7C", "2K3I8P", "9Y3U5X", "9Z0Q4K"]
                for anonymous_name in anonymous_list:
                    if anonymous_name not in cot_process:
                        cot_process[anonymous_name] = {}

                    # parse
                    visible_res = parse_visible_anonymous_data(anonymous_name=anonymous_name, use_history_img=args.enable_visible_img, streamline=cot_streamline)
                    
                    # Input_Memory
                    Input_Memory = {
                        "Step_0": {
                            "state": False,
                            "sub_step": None
                        },
                        "Step_1": {
                            "state": False,
                            "sub_step": (visible_res['profile']['input'], visible_res['profile']['images']) # profile
                        },
                        "Step_2": {
                            "state": False,
                            "sub_step": (visible_res['history']['input'], visible_res['history']['images']), # history
                        },
                        "Step_3": {
                            "state": False,
                            "sub_step": (visible_res['target']['input'], visible_res['target']['images']) # target
                        },
                        "Step_4": {
                            "state": False,
                            "sub_step": None
                        },
                        "Step_5": {
                            "state": False,
                            "sub_step": None
                        },
                        "Step_6": {
                            "state": False,
                            "sub_step": None
                        },
                        "Step_7": {
                            "state": False,
                            "sub_step": None
                        },
                        "Step_8": {
                            "state": False,
                            "sub_step": None
                        },
                        "Step_9": {
                            "state": False,
                            "sub_step": None
                        },
                        "Step_10": {
                            "state": False,
                            "sub_step": None
                        }
                    }
                    data = {'history': []}
                    inherient_flag = False # 是否开始继承优化标识
                    
                    # cot iter
                    for cot_step, cot_info in Personalized_CoTs.items():
                        
                        prompt = cot_info["prompt"]
                        accept_list = cot_info["accept"]
                        keep_memory = cot_info["keep_memory"]
                        is_inherient = cot_info["inherient"]

                        # 继承模式下直接跳过CoT生成步骤
                        if inherient_gen_data is not None and not is_inherient and not inherient_flag:
                            continue
                        # 继承模式下获得待优化标题，并标记已经开始优化
                        elif inherient_gen_data is not None and is_inherient:
                            inherient_flag = True # 标记已经开始进行优化
                            inherient_title = inherient_gen_data[anonymous_name] # 待优化标题
                        # 非继承模式下遇见优化步骤时的待优化标题初始化
                        elif inherient_gen_data is None and is_inherient:
                            inherient_title = cot_process[anonymous_name]["Step_4"]
                        
                        # 根据keep_memory重置data['history']
                        if not keep_memory:
                            data['history'] = []
                            data['images'] = []

                        step_end_license = False # 当前cot步骤结束许可
                        # 只要未拿到cot步骤结束许可，一直保持循环
                        while not step_end_license:
                            # Step Prompt尚未执行
                            if not Input_Memory[cot_step]['state']:
                                # Input_Memory中sub_step为None的模型只需要执行一次
                                if Input_Memory[cot_step]["sub_step"] is None:
                                    step_end_license = True # 允许在本次推理后步入next step    
                                Input_Memory[cot_step]['state'] = True

                                # ---------- 对后续优化prompt的处理 ------------ # 

                                if cot_step == "Step_5":
                                    
                             
                                    prompt = """你将扮演文本润色助手。我将给你提供某用户的历史发帖标题作为个性参考。并给你提供该用户的最新发帖图像，和我为该最新图像设计的标题，请你严格遵照以下三步对我设计的标题进行保留或润色。
                                            1. 历史标题符号使用格式检查
                                            2. 标题复杂性检查
                                            3. 标题个性化检查
                                            现在，让我们开始吧！
                                    """
                                        
                                    img_list = []

                                    # 当前匿名数据
                                    current_profile, _ = parse_visible_profile(anonymous_name=anonymous_name, streamline=True)
                                    current_history, _ = parse_visible_history(anonymous_name=anonymous_name, use_history_img=False, streamline=True)
                                    _, current_img_list = parse_visible_target(anonymous_name=anonymous_name, history_img_list=[], streamline=True)

                                    prompt = prompt + current_history + f"该用户的最新发帖图像为：<image>，我设计的标题为：{inherient_title}，请你严格遵循上述三步对我设计的标题进行润色或保留："
                                    img_list = img_list + current_img_list
                                    data["images"] = current_img_list

                                    # import pdb; pdb.set_trace()

                                # --------------------------------------------- # 

                                # 最终prompt为本次的query
                                query = prompt
                            # Step prompt已经执行
                            else:
                                query, img_list = Input_Memory[cot_step]['sub_step']
                                step_end_license = True # 允许在本次推理后步入next step
                                # 填补data
                                if len(img_list) > 0:
                                    data['images'] = img_list.copy()


                            # 构造推理kwargs
                            kwargs = {'query': prompt}
                            # 获取data参数
                            history = data.get('history')
                            system = data.get('system')
                            tools = data.get('tools')
                            objects = data.get('objects')
                            # 如果采用了详尽输出且system不为None，打印system
                            if args.verbose and system is not None:
                                print(f'[SYSTEM]{system}')
                            # ------处理历史对话------
                            if history is None:
                                history = []
                            # 加入kwargs字典
                            kwargs['history'] = history
                            # ------处理system------
                            if system is None and template.use_default_system:
                                system = template.default_system
                            # 加入kwargs字典
                            kwargs['system'] = system
                            # ------处理多媒体文件------
                            # 遍历media_key: [images, audios, videos]
                            for media_key in MediaTag.media_keys.values():
                                media_files = data.get(media_key)
                                # 加入kwargs字典
                                if media_files is not None:
                                    kwargs[media_key] = media_files
                            # ------处理tools、objects、truncation_strategy------
                            if tools is not None:
                                kwargs['tools'] = tools
                            if objects is not None:
                                kwargs['objects'] = objects
                            kwargs['truncation_strategy'] = args.truncation_strategy
                            # 如果推理后端为{'vllm', 'lmdeploy'}
                            if args.infer_backend in {'vllm', 'lmdeploy'}:
                                assert args.stream
                                if args.verbose:
                                    print(f"[QUERY]{data['query']}\n[RESPONSE]", end='')
                                gen = inference_stream_x(llm_engine, template, [kwargs], lora_request=lora_request)
                                print_idx = 0
                                for resp_list in gen:
                                    response = resp_list[0]['response']
                                    if args.verbose and len(response) > print_idx:
                                        print(response[print_idx:], end='', flush=True)
                                        print_idx = len(response)
                                print()
                            # 调用inference接口得到response
                            else:
                                response, new_history = inference(
                                    model, template, stream=args.stream and args.verbose, verbose=args.verbose, **kwargs)
                                
                            # 如果keep_memory为True，则更新记忆
                            if keep_memory:
                                data['history'] = new_history

                            # 记录response
                            cot_process[anonymous_name][cot_step] = response
                            
                            # 若data中包含了标签信息，赋值为label
                            label = data.pop('response', None)
                            # 构建对话保存实例
                            obj = {
                                'system': kwargs['system'],
                                'query': kwargs['query'],
                                'response': response,
                                'label': label,
                                'history': kwargs['history'],
                            }
                            # 将多媒体文件路径加入对话保存实例
                            for media_key in MediaTag.media_keys.values():
                                media_files = kwargs.get(media_key)
                                if media_files is not None:
                                    obj[media_key] = media_files
                            # 输出到对话存储文件中
                            if jsonl_path is not None:
                                append_to_jsonl(jsonl_path, obj)
                            # 将对话保存实例加入result中
                            result.append(obj)

                # save
                unique_name = f"{args.model_type}-{control_memory_bank}-{args.cot_mode}-{int(args.use_cot_history_img)}-{int(args.cot_streamline)}"
                unique_index = 0
                while(os.path.exists(os.path.join(args.cot_save_dir, f"{unique_name}#{unique_index}.json"))):
                    unique_index += 1
                cot_save_path = os.path.join(args.cot_save_dir, f"{unique_name}#{unique_index}.json")
                with open(cot_save_path, 'w', encoding='utf-8') as f:
                    json.dump(cot_process, f, ensure_ascii=False, indent=4)
                print('----------------COT推理完成-----------------')


            elif args.cot_mode == 'dataset':
                for group_name, group_value in cot_dataset.items():
                    group_anonymous_list = [name for name in group_value if name != 'count']
                    for anonymous_name in group_anonymous_list:
                        if anonymous_name not in cot_process:
                            cot_process[anonymous_name] = {}

                        # parse
                        visible_res = parse_visible_anonymous_data(anonymous_name=anonymous_name, use_history_img=args.enable_visible_img, streamline=cot_streamline)
                        
                        # Input_Memory
                        Input_Memory = {
                            "Step_0": {
                                "state": False,
                                "sub_step": None
                            },
                            "Step_1": {
                                "state": False,
                                "sub_step": (visible_res['profile']['input'], visible_res['profile']['images']) # profile
                            },
                            "Step_2": {
                                "state": False,
                                "sub_step": (visible_res['history']['input'], visible_res['history']['images']), # history
                            },
                            "Step_3": {
                                "state": False,
                                "sub_step": (visible_res['target']['input'], visible_res['target']['images']) # target
                            },
                            "Step_4": {
                                "state": False,
                                "sub_step": None
                            },
                            "Step_5": {
                                "state": False,
                                "sub_step": None
                            }
                        }
                        data = {'history': []}
                        inherient_flag = False # 是否开始继承优化标识
                        
                        # cot iter
                        for cot_step, cot_info in Personalized_CoTs.items():
                            
                            prompt = cot_info["prompt"]
                            accept_list = cot_info["accept"]
                            keep_memory = cot_info["keep_memory"]
                            is_inherient = cot_info["inherient"]

                            # 继承模式下直接跳过CoT生成步骤
                            if inherient_gen_data is not None and not is_inherient and not inherient_flag:
                                continue
                            # 继承模式下获得待优化标题，并标记已经开始优化
                            elif inherient_gen_data is not None and is_inherient:
                                inherient_flag = True # 标记已经开始进行优化
                                inherient_title = inherient_gen_data[anonymous_name] # 待优化标题
                            # 非继承模式下遇见优化步骤时的待优化标题初始化
                            elif inherient_gen_data is None and is_inherient:
                                inherient_title = cot_process[anonymous_name]["Step_4"]
                            
                            # 根据keep_memory重置data['history']
                            if not keep_memory:
                                data['history'] = []
                                data['images'] = []

                            step_end_license = False # 当前cot步骤结束许可
                            # 只要未拿到cot步骤结束许可，一直保持循环
                            while not step_end_license:
                                # Step Prompt尚未执行
                                if not Input_Memory[cot_step]['state']:
                                    # Input_Memory中sub_step为None的模型只需要执行一次
                                    if Input_Memory[cot_step]["sub_step"] is None:
                                        step_end_license = True # 允许在本次推理后步入next step    
                                    Input_Memory[cot_step]['state'] = True

                                    # ---------- 对后续优化prompt的处理 ------------ # 

                                    if cot_step == "Step_5":
                                        
                                        
                                        prompt = """你将扮演文本润色助手。我将给你提供某用户的历史发帖标题作为个性参考。并给你提供该用户的最新发帖图像，和我为该最新图像设计的标题，请你严格遵照以下三步对我设计的标题进行保留或润色。
                                                1. 历史标题符号使用格式检查
                                                2. 标题复杂性检查
                                                3. 标题个性化检查
                                                现在，让我们开始吧！
                                        """
                                              
                                        img_list = []

                                        # 当前匿名数据
                                        current_profile, _ = parse_visible_profile(anonymous_name=anonymous_name, streamline=True)
                                        current_history, _ = parse_visible_history(anonymous_name=anonymous_name, use_history_img=False, streamline=True)
                                        _, current_img_list = parse_visible_target(anonymous_name=anonymous_name, history_img_list=[], streamline=True)
      
                                        prompt = prompt + current_history + f"该用户的最新发帖图像为：<image>，我设计的标题为：{inherient_title}，请你严格遵循上述三步对我设计的标题进行润色或保留："
                                        img_list = img_list + current_img_list
                                        data["images"] = current_img_list

                                        # import pdb; pdb.set_trace()

                                    # --------------------------------------------- # 

                                    # 最终prompt为本次的query
                                    query = prompt
                                # Step prompt已经执行
                                else:
                                    query, img_list = Input_Memory[cot_step]['sub_step']
                                    step_end_license = True # 允许在本次推理后步入next step
                                    # 填补data
                                    if len(img_list) > 0:
                                        data['images'] = img_list.copy()

                                # 构造推理kwargs
                                kwargs = {'query': prompt}
                                # 获取data参数
                                history = data.get('history')
                                system = data.get('system')
                                tools = data.get('tools')
                                objects = data.get('objects')
                                # 如果采用了详尽输出且system不为None，打印system
                                if args.verbose and system is not None:
                                    print(f'[SYSTEM]{system}')
                                # ------处理历史对话------
                                if history is None:
                                    history = []
                                # 加入kwargs字典
                                kwargs['history'] = history
                                # ------处理system------
                                if system is None and template.use_default_system:
                                    system = template.default_system
                                # 加入kwargs字典
                                kwargs['system'] = system
                                # ------处理多媒体文件------
                                # 遍历media_key: [images, audios, videos]
                                for media_key in MediaTag.media_keys.values():
                                    media_files = data.get(media_key)
                                    # 加入kwargs字典
                                    if media_files is not None:
                                        kwargs[media_key] = media_files
                                # ------处理tools、objects、truncation_strategy------
                                if tools is not None:
                                    kwargs['tools'] = tools
                                if objects is not None:
                                    kwargs['objects'] = objects
                                kwargs['truncation_strategy'] = args.truncation_strategy
                                # 如果推理后端为{'vllm', 'lmdeploy'}
                                if args.infer_backend in {'vllm', 'lmdeploy'}:
                                    assert args.stream
                                    if args.verbose:
                                        print(f"[QUERY]{data['query']}\n[RESPONSE]", end='')
                                    gen = inference_stream_x(llm_engine, template, [kwargs], lora_request=lora_request)
                                    print_idx = 0
                                    for resp_list in gen:
                                        response = resp_list[0]['response']
                                        if args.verbose and len(response) > print_idx:
                                            print(response[print_idx:], end='', flush=True)
                                            print_idx = len(response)
                                    print()
                                # 调用inference接口得到response
                                else:
                                    response, new_history = inference(
                                        model, template, stream=args.stream and args.verbose, verbose=args.verbose, **kwargs)
                                    
                                # 如果keep_memory为True，则更新记忆
                                if keep_memory:
                                    data['history'] = new_history

                                # 记录response
                                cot_process[anonymous_name][cot_step] = response
                                
                                # 若data中包含了标签信息，赋值为label
                                label = data.pop('response', None)
                                # 构建对话保存实例
                                obj = {
                                    'system': kwargs['system'],
                                    'query': kwargs['query'],
                                    'response': response,
                                    'label': label,
                                    'history': kwargs['history'],
                                }
                                # 将多媒体文件路径加入对话保存实例
                                for media_key in MediaTag.media_keys.values():
                                    media_files = kwargs.get(media_key)
                                    if media_files is not None:
                                        obj[media_key] = media_files
                                # 输出到对话存储文件中
                                if jsonl_path is not None:
                                    append_to_jsonl(jsonl_path, obj)
                                # 将对话保存实例加入result中
                                result.append(obj)

                # save
                unique_name = f"{args.model_type}-{control_memory_bank}-{args.cot_mode}-{int(args.use_cot_history_img)}-{int(args.cot_streamline)}"
                unique_index = 0
                while(os.path.exists(os.path.join(args.cot_save_dir, f"{unique_name}#{unique_index}.json"))):
                    unique_index += 1
                cot_save_path = os.path.join(args.cot_save_dir, f"{unique_name}#{unique_index}.json")
                with open(cot_save_path, 'w', encoding='utf-8') as f:
                    json.dump(cot_process, f, ensure_ascii=False, indent=4)
                print('----------------COT推理完成-----------------')


            else:
                raise ValueError(f"Your provided cot-mode: {args.cot_mode} doesn't meet requirement. ")


        # 执行直接记忆版本
        else:
           

            Personalized_CoTs = {
                "Step_0": "你将扮演多媒体平台智能标题生成助手，根据博主的个人简介、历史发帖（包括发帖图像及对应标题），为其生成个性化的标题。请严格按照以下3个步骤进行思考和生成。",
                "Step_1": "提取个性和情绪信息：从博主的个人简介中提取关键的个性和情绪特征，包括职业、兴趣爱好、表达倾向以及整体情绪风格。请特别注意情绪风格的细节描述，明确博主是否呈现出积极、消极、幽默、严肃等情感倾向，并简洁明了地总结这些信息。输出格式为：兴趣爱好：XX，表达倾向：XXX，情绪风格：XXX（例如：幽默、轻松、严肃）。",
                "Step_2": "分析文字和符号使用习惯：分析该博主历史发帖标题中的文字和符号表达习惯，重点分析以下几点：1. 是否存在固定的标题结构（如主副标题分隔、并列内容等）。2. 是否偏好使用特定符号（如“|”或“/”等），以及这些符号的功能和使用场景（例如：分隔不同主题、突出重点等）。3. 标题的总体风格：文学化、描述性或现代化表达。请特别注意历史标题中的分隔符和结构，并在生成标题时严格遵循相同的格式。如果历史标题中使用了分隔符，请详细识别这些分隔符的类型及其功能，并确保生成标题时保持一致性。",
                "Step_3": "生成个性化标题：根据上述分析结果（个人简介、历史发帖习惯），我将提供该博主的最新发帖图像，请你为该最新发帖图像生成严格遵循博主分隔符偏好的个性化标题。",
                # "Step_4": "自我校验和改写：请根据你生成的标题，在遵循博主分隔符偏好的基础上，从不同角度对其进行改写，并选择你认为最合适的一个。输出格式为：最终版本：XXX"
            }    


            '''
            Personalized_CoTs = {
                "Step_0": "你将扮演多媒体平台智能标题生成助手，根据博主的历史发帖（包括发帖图像及对应标题），为其生成个性化的标题。请严格按照以下2个步骤进行思考和生成。",
                "Step_1": "分析文字和符号使用习惯：分析该博主历史发帖标题中的文字和符号表达习惯，重点分析以下几点：1. 是否存在固定的标题结构（如主副标题分隔、并列内容等）。2. 是否偏好使用特定符号（如“|”或“/”等），以及这些符号的功能和使用场景（例如：分隔不同主题、突出重点等）。3. 标题的总体风格：文学化、描述性或现代化表达。请特别注意历史标题中的分隔符和结构，并在生成标题时严格遵循相同的格式。如果历史标题中使用了分隔符，请详细识别这些分隔符的类型及其功能，并确保生成标题时保持一致性。",
                "Step_2": "生成个性化标题：根据上述分析结果（历史发帖习惯），我将提供该博主的最新发帖图像，请你为该最新发帖图像生成严格遵循博主分隔符偏好的个性化标题。",
                # "Step_4": "自我校验和改写：请根据你生成的标题，在遵循博主分隔符偏好的基础上，从不同角度对其进行改写，并选择你认为最合适的一个。输出格式为：最终版本：XXX"
            }    
            '''

            # parse function
            def parse_profile(anonymous_name: str, streamline=False):
                '''
                profile
                '''

                if anonymous_name in group_data:
                    group_index = group_data[anonymous_name]
                    anonymous_data = cot_dataset[group_index][anonymous_name]
                else:
                    anonymous_data = cot_dataset[anonymous_name]
                user = anonymous_data['user']
                profile_input = "该博主的个人简介为：性别：{}; 职业: {}; 自我描述：{}".format(user['sex'], user['job'], user['desc_info'])

                return (profile_input, [])

            def parse_history(anonymous_name: str, use_history_img=False, streamline=False):
                '''
                history
                '''
                if anonymous_name in group_data:
                    group_index = group_data[anonymous_name]
                    anonymous_data = cot_dataset[group_index][anonymous_name]
                else:
                    anonymous_data = cot_dataset[anonymous_name]
                history_list = anonymous_data['history']
                img_list = []
                history_input = "该博主的发帖历史为："
                # 使用原始图像
                if not streamline:
                    for i, history_note in enumerate(history_list):
                        history_input = history_input + f"{i+1}. "
                        if use_history_img:
                            history_input = history_input + f"标题：{history_note['title']}，对应图像：{len(history_note['image_urls']) * '<image>'}"
                        else:
                            history_input = history_input + f"标题：{history_note['title']}"
                        if i + 1 == len(history_list):
                            history_input = history_input + "。 "
                        else:
                            history_input = history_input + "; "
                        if use_history_img:
                            img_list = img_list + history_note['image_urls']
                # 使用拼接图像
                else:
                    for i, history_note in enumerate(history_list):
                        history_input = history_input + f"{i+1}. "
                        if use_history_img:
                            history_input = history_input + f"标题：{history_note['title']}，对应图像：<image>"
                        else:
                            history_input = history_input + f"标题：{history_note['title']}"
                        if i + 1 == len(history_list):
                            history_input = history_input + "。 "
                        else:
                            history_input = history_input + "; "
                        if use_history_img:
                            if "sample_prefix" in history_note:
                                img_list.append(os.path.join(img_dir, history_note['sample_prefix']))
                            elif "img_prefix" in history_note:
                                img_list.append(os.path.join(img_dir, history_note['img_prefix']))
                            else:
                                raise ValueError("你提供的数据集中不存在图像字段")
                return (history_input, img_list)

            def parse_target(anonymous_name: str, history_img_list: list, streamline=False):
                if anonymous_name in group_data:
                    group_index = group_data[anonymous_name]
                    anonymous_data = cot_dataset[group_index][anonymous_name]
                else:
                    anonymous_data = cot_dataset[anonymous_name]
                target_note = anonymous_data['target']
                # 使用原始图像
                if not streamline:
                    img_list = target_note['image_urls']
                    target_input = f"该博主的最新发帖图像为：{len(img_list) * '<image>'}，请执行生成个性化标题任务！直接输出标题即可。输出格式为：最终版本：XXX"
                    img_list = history_img_list + img_list
                # 使用拼接图像
                else:
                    if "sample_prefix" in target_note:
                        img_list = [os.path.join(img_dir, target_note['sample_prefix'])]
                    elif "img_prefix" in target_note:
                        img_list = [(os.path.join(img_dir, target_note['img_prefix']))]
                    else:
                        raise ValueError("你提供的数据集中不存在图像字段")
                    img_list = history_img_list + img_list
                    target_input = f"该博主的最新发帖图像为：<image>，请执行生成个性化标题任务！直接输出标题即可。你生成的个性化标题为："
                return (target_input, img_list)

            def parse_anonymous_data(anonymous_name, use_history_img=False, streamline=False):
                
                parse_res = {}
                # profile
                profile_input, _ = parse_profile(anonymous_name=anonymous_name, streamline=streamline)
                parse_res['profile'] = {'input': profile_input, 'images': []}
                # history
                history_input, history_img_list = parse_history(anonymous_name=anonymous_name, use_history_img=use_history_img, streamline=streamline)
                parse_res['history'] = {'input': history_input, 'images': history_img_list}
                # target
                target_input, target_img_list = parse_target(anonymous_name=anonymous_name, history_img_list=history_img_list, streamline=streamline)
                parse_res['target'] = {'input': target_input, 'images': target_img_list}

                return parse_res

            # 记录最终生成回复
            cot_res = {}

            # run item
            if args.cot_mode == 'test':
                anonymous_list = ['4L2G2Z', '5O6O0C', '7Q1K1Y', '2L3C1N', '2N3P3H', '8B6L5D', '0F9S3M', '6L0X6X', '6J1K5J', '3Q8X4Y', '9C8P0H', '9I3H7B', '0U6O3E', '3Y2P2B', '1E5I7E', '2L5W6D', '0F4L5L', '0W0H2T', '0G2Y7M', '8J6L2D', '7O2D0C', '5V2G2U', '4N8T9Q', '0D9N5X', '2Y4T4A', '9N8S6G', '5T4Z8M', '1U9V8Z', '1P2B4H', '1O8H6C', '6S1H9N', '8I6Y7E', '6H0U3Y', '2C5I0B', '7G6G5W', '9S1K1I']
                for anonymous_name in anonymous_list:
                    if anonymous_name not in cot_res:
                        cot_res[anonymous_name] = []

                    # parse
                    parse_res = parse_anonymous_data(anonymous_name=anonymous_name, use_history_img=use_cot_history_img, streamline=cot_streamline)
                    # data = {'system': 'You are an intelligent assistant for generating personalized titles.', 'history': []}
                    data = {'history': []}
                    '''
                    Input_Memory = {
                        "Step_0": {
                            "state": False,
                            "accept": None
                        },
                        "Step_1": {
                            "state": False,
                            "accept": (parse_res['profile']['input'], parse_res['profile']['images']) # profile
                        },
                        "Step_2": {
                            "state": False,
                            "accept": (parse_res['history']['input'], parse_res['history']['images']), # history
                        },
                        "Step_3": {
                            "state": False,
                            "accept": (parse_res['target']['input'], parse_res['target']['images']) # target
                        },
                        "Step_4": {
                            "state": False,
                            "accept": None
                        }
                    }
                    '''
                    Input_Memory = {
                        "Step_0": {
                            "state": False,
                            "accept": None
                        },
                        "Step_1": {
                            "state": False,
                            "accept": (parse_res['history']['input'], parse_res['history']['images']), # history
                        },
                        "Step_2": {
                            "state": False,
                            "accept": (parse_res['target']['input'], parse_res['target']['images']) # target
                        },
                        "Step_3": {
                            "state": False,
                            "accept": None
                        }
                    }
                    
                    # cot iter
                    for cot_step, cot_prompt in Personalized_CoTs.items():
                        
                        step_end_license = False # 当前cot步骤结束许可
                        while not step_end_license:
                            # Step Prompt尚未执行
                            if not Input_Memory[cot_step]['state']:
                                if cot_step == "Step_0" or cot_step == "Step_4":
                                    query = cot_prompt
                                    step_end_license = True # 允许在本次推理后步入next step
                                else:
                                    query = cot_step + ":" + cot_prompt
                                Input_Memory[cot_step]['state'] = True

                            # Step prompt已经执行
                            else:
                                query, img_list = Input_Memory[cot_step]['accept']
                                step_end_license = True # 允许在本次推理后步入next step
                                # 填补data
                                if len(img_list) > 0:
                                    data['images'] = img_list.copy()
                                
                            # 构造推理kwargs
                            kwargs = {'query': query}
                        
                            # 获取data参数
                            history = data.get('history')
                            system = data.get('system')
                            tools = data.get('tools')
                            objects = data.get('objects')
                            # 如果采用了详尽输出且system不为None，打印system
                            if args.verbose and system is not None:
                                print(f'[SYSTEM]{system}')
                            # ------处理历史对话------
                            if history is None:
                                history = []
                            # 加入kwargs字典
                            kwargs['history'] = history
                            # ------处理system------
                            if system is None and template.use_default_system:
                                system = template.default_system
                            # 加入kwargs字典
                            kwargs['system'] = system
                            # ------处理多媒体文件------
                            # 遍历media_key: [images, audios, videos]
                            for media_key in MediaTag.media_keys.values():
                                media_files = data.get(media_key)
                                # 加入kwargs字典
                                if media_files is not None:
                                    kwargs[media_key] = media_files
                            # ------处理tools、objects、truncation_strategy------
                            if tools is not None:
                                kwargs['tools'] = tools
                            if objects is not None:
                                kwargs['objects'] = objects
                            kwargs['truncation_strategy'] = args.truncation_strategy
                            # 如果推理后端为{'vllm', 'lmdeploy'}
                            if args.infer_backend in {'vllm', 'lmdeploy'}:
                                assert args.stream
                                if args.verbose:
                                    print(f"[QUERY]{data['query']}\n[RESPONSE]", end='')
                                gen = inference_stream_x(llm_engine, template, [kwargs], lora_request=lora_request)
                                print_idx = 0
                                for resp_list in gen:
                                    response = resp_list[0]['response']
                                    if args.verbose and len(response) > print_idx:
                                        print(response[print_idx:], end='', flush=True)
                                        print_idx = len(response)
                                print()
                            # 调用inference接口得到response
                            else:
                                response, new_history = inference(
                                    model, template, stream=args.stream and args.verbose, verbose=args.verbose, **kwargs)
                                
                            # 更新history
                            data['history'] = new_history

                            # 记录response
                            cot_res[anonymous_name].append(response)
                            
                            # 若data中包含了标签信息，赋值为label
                            label = data.pop('response', None)
                            # 构建对话保存实例
                            obj = {
                                'system': kwargs['system'],
                                'query': kwargs['query'],
                                'response': response,
                                'label': label,
                                'history': kwargs['history'],
                            }
                            # 将多媒体文件路径加入对话保存实例
                            for media_key in MediaTag.media_keys.values():
                                media_files = kwargs.get(media_key)
                                if media_files is not None:
                                    obj[media_key] = media_files
                            # 输出到对话存储文件中
                            if jsonl_path is not None:
                                append_to_jsonl(jsonl_path, obj)
                            # 将对话保存实例加入result中
                            result.append(obj)

                # save
                unique_name = f"{args.model_type}-{control_memory_bank}-{args.cot_mode}-{int(args.use_cot_history_img)}-{int(args.cot_streamline)}"
                unique_index = 0
                while(os.path.exists(os.path.join(args.cot_save_dir, f"{unique_name}#{unique_index}.json"))):
                    unique_index += 1
                cot_save_path = os.path.join(args.cot_save_dir, f"{unique_name}#{unique_index}.json")
                with open(cot_save_path, 'w', encoding='utf-8') as f:
                    json.dump(cot_res, f, ensure_ascii=False, indent=4)
                print('----------------COT推理完成-----------------')


            # run dataset
            elif args.cot_mode == 'dataset':
                if args.is_sample:
                    for group_name, group_value in tqdm(cot_dataset.items()):
                        group_anonymous_list = [name for name in group_value if name != 'count']
                        for anonymous_name in group_anonymous_list:
                            if anonymous_name not in cot_res:
                                cot_res[anonymous_name] = []
                            # parse
                            parse_res = parse_anonymous_data(anonymous_name=anonymous_name, use_history_img=use_cot_history_img, streamline=cot_streamline)
                            # data = {'system': 'You are an intelligent assistant for generating personalized titles.', 'history': []}
                            data = {'history': []}
                            
                            Input_Memory = {
                                "Step_0": {
                                    "state": False,
                                    "accept": None
                                },
                                "Step_1": {
                                    "state": False,
                                    "accept": (parse_res['profile']['input'], parse_res['profile']['images']) # profile
                                },
                                "Step_2": {
                                    "state": False,
                                    "accept": (parse_res['history']['input'], parse_res['history']['images']), # history
                                },
                                "Step_3": {
                                    "state": False,
                                    "accept": (parse_res['target']['input'], parse_res['target']['images']) # target
                                },
                                "Step_4": {
                                    "state": False,
                                    "accept": None
                                }
                            }
                            '''
                            Input_Memory = {
                                "Step_0": {
                                    "state": False,
                                    "accept": None
                                },
                                "Step_1": {
                                    "state": False,
                                    "accept": (parse_res['history']['input'], parse_res['history']['images']), # history
                                },
                                "Step_2": {
                                    "state": False,
                                    "accept": (parse_res['target']['input'], parse_res['target']['images']) # target
                                },
                                "Step_3": {
                                    "state": False,
                                    "accept": None
                                }
                            }
                            '''
                            # cot iter
                            for cot_step, cot_prompt in Personalized_CoTs.items():
                                step_end_license = False # 当前cot步骤结束许可
                                while not step_end_license:
                                    # Step Prompt尚未执行
                                    if not Input_Memory[cot_step]['state']:
                                        if cot_step == "Step_0" or cot_step == "Step_4":
                                            query = cot_prompt
                                            step_end_license = True # 允许在本次推理后步入next step
                                        else:
                                            query = cot_step + ":" + cot_prompt
                                        Input_Memory[cot_step]['state'] = True

                                    # Step prompt已经执行
                                    else:
                                        query, img_list = Input_Memory[cot_step]['accept']
                                        step_end_license = True # 允许在本次推理后步入next step
                                        # 填补data
                                        if len(img_list) > 0:
                                            data['images'] = img_list.copy()
                                        
                                    # 构造推理kwargs
                                    kwargs = {'query': query}
                                
                                    # 获取data参数
                                    history = data.get('history')
                                    system = data.get('system')
                                    tools = data.get('tools')
                                    objects = data.get('objects')
                                    # 如果采用了详尽输出且system不为None，打印system
                                    if args.verbose and system is not None:
                                        print(f'[SYSTEM]{system}')
                                    # ------处理历史对话------
                                    if history is None:
                                        history = []
                                    # 加入kwargs字典
                                    kwargs['history'] = history
                                    # ------处理system------
                                    if system is None and template.use_default_system:
                                        system = template.default_system
                                    # 加入kwargs字典
                                    kwargs['system'] = system
                                    # ------处理多媒体文件------
                                    # 遍历media_key: [images, audios, videos]
                                    for media_key in MediaTag.media_keys.values():
                                        media_files = data.get(media_key)
                                        # 加入kwargs字典
                                        if media_files is not None:
                                            kwargs[media_key] = media_files
                                    # ------处理tools、objects、truncation_strategy------
                                    if tools is not None:
                                        kwargs['tools'] = tools
                                    if objects is not None:
                                        kwargs['objects'] = objects
                                    kwargs['truncation_strategy'] = args.truncation_strategy
                                    # 如果推理后端为{'vllm', 'lmdeploy'}
                                    if args.infer_backend in {'vllm', 'lmdeploy'}:
                                        assert args.stream
                                        if args.verbose:
                                            print(f"[QUERY]{data['query']}\n[RESPONSE]", end='')
                                        gen = inference_stream_x(llm_engine, template, [kwargs], lora_request=lora_request)
                                        print_idx = 0
                                        for resp_list in gen:
                                            response = resp_list[0]['response']
                                            if args.verbose and len(response) > print_idx:
                                                print(response[print_idx:], end='', flush=True)
                                                print_idx = len(response)
                                        print()
                                    # 调用inference接口得到response
                                    else:
                                        response, new_history = inference(
                                            model, template, stream=args.stream and args.verbose, verbose=args.verbose, **kwargs)
                                        
                                    # 更新history
                                    data['history'] = new_history

                                    # 加入cot_res
                                    cot_res[anonymous_name].append(response)
                                    
                                    # 若data中包含了标签信息，赋值为label
                                    label = data.pop('response', None)
                                    # 构建对话保存实例
                                    obj = {
                                        'system': kwargs['system'],
                                        'query': kwargs['query'],
                                        'response': response,
                                        'label': label,
                                        'history': kwargs['history'],
                                    }
                                    # 将多媒体文件路径加入对话保存实例
                                    for media_key in MediaTag.media_keys.values():
                                        media_files = kwargs.get(media_key)
                                        if media_files is not None:
                                            obj[media_key] = media_files
                                    # 输出到对话存储文件中
                                    if jsonl_path is not None:
                                        append_to_jsonl(jsonl_path, obj)
                                    # 将对话保存实例加入result中
                                    result.append(obj)
                else:
                    
                    for anonymous_name in tqdm(cot_dataset):
                        if anonymous_name not in cot_res:
                            cot_res[anonymous_name] = []
                        # parse
                        parse_res = parse_anonymous_data(anonymous_name=anonymous_name, use_history_img=use_cot_history_img, streamline=cot_streamline)
                        # data = {'system': 'You are an intelligent assistant for generating personalized titles.', 'history': []}
                        data = {'history': []}

                        Input_Memory = {
                            "Step_0": {
                                "state": False,
                                "accept": None
                            },
                            "Step_1": {
                                "state": False,
                                "accept": (parse_res['profile']['input'], parse_res['profile']['images']) # profile
                            },
                            "Step_2": {
                                "state": False,
                                "accept": (parse_res['history']['input'], parse_res['history']['images']), # history
                            },
                            "Step_3": {
                                "state": False,
                                "accept": (parse_res['target']['input'], parse_res['target']['images']) # target
                            },
                            "Step_4": {
                                "state": False,
                                "accept": None
                            }
                        }
                        '''
                        Input_Memory = {
                            "Step_0": {
                                "state": False,
                                "accept": None
                            },
                            "Step_1": {
                                "state": False,
                                "accept": (parse_res['history']['input'], parse_res['history']['images']), # history
                            },
                            "Step_2": {
                                "state": False,
                                "accept": (parse_res['target']['input'], parse_res['target']['images']) # target
                            },
                            "Step_3": {
                                "state": False,
                                "accept": None
                            }
                        }
                        '''
                        # cot iter
                        for cot_step, cot_prompt in Personalized_CoTs.items():
                            step_end_license = False # 当前cot步骤结束许可
                            while not step_end_license:
                                # Step Prompt尚未执行
                                if not Input_Memory[cot_step]['state']:
                                    if cot_step == "Step_0" or cot_step == "Step_4":
                                        query = cot_prompt
                                        step_end_license = True # 允许在本次推理后步入next step
                                    else:
                                        query = cot_step + ":" + cot_prompt
                                    Input_Memory[cot_step]['state'] = True

                                # Step prompt已经执行
                                else:
                                    query, img_list = Input_Memory[cot_step]['accept']
                                    step_end_license = True # 允许在本次推理后步入next step
                                    # 填补data
                                    if len(img_list) > 0:
                                        data['images'] = img_list.copy()
                                    
                                # 构造推理kwargs
                                kwargs = {'query': query}
                            
                                # 获取data参数
                                history = data.get('history')
                                system = data.get('system')
                                tools = data.get('tools')
                                objects = data.get('objects')
                                # 如果采用了详尽输出且system不为None，打印system
                                if args.verbose and system is not None:
                                    print(f'[SYSTEM]{system}')
                                # ------处理历史对话------
                                if history is None:
                                    history = []
                                # 加入kwargs字典
                                kwargs['history'] = history
                                # ------处理system------
                                if system is None and template.use_default_system:
                                    system = template.default_system
                                # 加入kwargs字典
                                kwargs['system'] = system
                                # ------处理多媒体文件------
                                # 遍历media_key: [images, audios, videos]
                                for media_key in MediaTag.media_keys.values():
                                    media_files = data.get(media_key)
                                    # 加入kwargs字典
                                    if media_files is not None:
                                        kwargs[media_key] = media_files
                                # ------处理tools、objects、truncation_strategy------
                                if tools is not None:
                                    kwargs['tools'] = tools
                                if objects is not None:
                                    kwargs['objects'] = objects
                                kwargs['truncation_strategy'] = args.truncation_strategy
                                # 如果推理后端为{'vllm', 'lmdeploy'}
                                if args.infer_backend in {'vllm', 'lmdeploy'}:
                                    assert args.stream
                                    if args.verbose:
                                        print(f"[QUERY]{data['query']}\n[RESPONSE]", end='')
                                    gen = inference_stream_x(llm_engine, template, [kwargs], lora_request=lora_request)
                                    print_idx = 0
                                    for resp_list in gen:
                                        response = resp_list[0]['response']
                                        if args.verbose and len(response) > print_idx:
                                            print(response[print_idx:], end='', flush=True)
                                            print_idx = len(response)
                                    print()
                                # 调用inference接口得到response
                                else:
                                    response, new_history = inference(
                                        model, template, stream=args.stream and args.verbose, verbose=args.verbose, **kwargs)
                                    
                                # 更新history
                                data['history'] = new_history

                                # 加入cot_res
                                cot_res[anonymous_name].append(response)
                                
                                # 若data中包含了标签信息，赋值为label
                                label = data.pop('response', None)
                                # 构建对话保存实例
                                obj = {
                                    'system': kwargs['system'],
                                    'query': kwargs['query'],
                                    'response': response,
                                    'label': label,
                                    'history': kwargs['history'],
                                }
                                # 将多媒体文件路径加入对话保存实例
                                for media_key in MediaTag.media_keys.values():
                                    media_files = kwargs.get(media_key)
                                    if media_files is not None:
                                        obj[media_key] = media_files
                                # 输出到对话存储文件中
                                if jsonl_path is not None:
                                    append_to_jsonl(jsonl_path, obj)
                                # 将对话保存实例加入result中
                                result.append(obj)    

                # save
                unique_name = f"{args.model_type}-{control_memory_bank}-{args.cot_mode}-{int(args.use_cot_history_img)}-{int(args.cot_streamline)}"
                unique_index = 0
                while(os.path.exists(os.path.join(args.cot_save_dir, f"{unique_name}#{unique_index}.json"))):
                    unique_index += 1
                cot_save_path = os.path.join(args.cot_save_dir, f"{unique_name}#{unique_index}.json")
                with open(cot_save_path, 'w', encoding='utf-8') as f:
                    json.dump(cot_res, f, ensure_ascii=False, indent=4)
                print('----------------COT推理完成-----------------')

            else:
                raise ValueError(f"Your provided cot-mode: {args.cot_mode} doesn't meet requirement. ")

    elif args.enforce_transfer:
        '''
        转化GPT-4o评估文本
        '''
        if args.transfer_mode == 'test':
            pass

        elif args.transfer_mode == 'dataset':
            
            deque_files = os.listdir(args.transfer_deque_dir)
            for deque_file in deque_files:
                deque_path = os.path.join(args.transfer_deque_dir, deque_file)
                out_path = os.path.join(args.transfer_save_dir, deque_file)
                deque_data = load_json(deque_path)
                save_data = {}
                for anonymous_name, gpt_response in deque_data.items():

                    data = {}
                    query = "请你将以下文本的最终排序结果提取出来，组织为类似{{Rankings: model_A > model_H = model_C, Reasons: XXX}}的格式。文本内容为：" + gpt_response
                    # query = "请将以下文本内容中最终的润色标题提取出来，直接输出润色标题即可。文本内容为：{}".format(gpt_response)
                    # query = "请将以下文本的最终排序结果提取出来，提取为model_A > model_B 或 model_A = model_B 或 model_A > model_B，直接输出即可。文本内容为：" + gpt_response
                    # 构造推理kwargs
                    kwargs = {'query': query}
            
                    # 获取data参数
                    history = data.get('history')
                    system = data.get('system')
                    # import pdb; pdb.set_trace()
                    tools = data.get('tools')
                    objects = data.get('objects')
                    # 如果采用了详尽输出且system不为None，打印system
                    if args.verbose and system is not None:
                        print(f'[SYSTEM]{system}')
                    # ------处理历史对话------
                    if history is None:
                        history = []
                    # 加入kwargs字典
                    kwargs['history'] = history
                    # ------处理system------
                    if system is None and template.use_default_system:
                        system = template.default_system
                    # 加入kwargs字典
                    kwargs['system'] = system
                    # ------处理多媒体文件------
                    # 遍历media_key: [images, audios, videos]
                    for media_key in MediaTag.media_keys.values():
                        media_files = data.get(media_key)
                        # 加入kwargs字典
                        if media_files is not None:
                            kwargs[media_key] = media_files
                    # ------处理tools、objects、truncation_strategy------
                    if tools is not None:
                        kwargs['tools'] = tools
                    if objects is not None:
                        kwargs['objects'] = objects
                    kwargs['truncation_strategy'] = args.truncation_strategy
                    if args.infer_backend in {'vllm', 'lmdeploy'}:
                        assert args.stream
                        if args.verbose:
                            print(f"[QUERY]{data['query']}\n[RESPONSE]", end='')
                        gen = inference_stream_x(llm_engine, template, [kwargs], lora_request=lora_request)
                        print_idx = 0
                        for resp_list in gen:
                            response = resp_list[0]['response']
                            if args.verbose and len(response) > print_idx:
                                print(response[print_idx:], end='', flush=True)
                                print_idx = len(response)
                        print()
                    # 调用inference接口得到response
                    else:
                        response, _ = inference(
                            model, template, stream=args.stream and args.verbose, verbose=args.verbose, **kwargs)
                    # 若data中包含了标签信息，赋值为label
                    label = data.pop('response', None)
                    # 构建对话保存实例
                    obj = {
                        'system': kwargs['system'],
                        'query': kwargs['query'],
                        'response': response,
                        'label': label,
                        'history': kwargs['history'],
                    }
                    # 将多媒体文件路径加入对话保存实例
                    for media_key in MediaTag.media_keys.values():
                        media_files = kwargs.get(media_key)
                        if media_files is not None:
                            obj[media_key] = media_files
                    # 输出到对话存储文件中
                    if jsonl_path is not None:
                        append_to_jsonl(jsonl_path, obj)
                    # 将对话保存实例加入result中
                    result.append(obj)

                    # 将回复加入save_data
                    save_data[anonymous_name] = response

                    if args.verbose:
                        print()
                        print(f'[LABELS]{label}')
                        for media_key in MediaTag.media_keys.values():
                            media_files = kwargs.get(media_key)
                            if media_files is not None:
                                print(f'[{media_key.upper()}]{media_files}')
                        print('-' * 50, flush=True)
                # save
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=4)

        else:
            raise ValueError(f'你提供的transfer_mode: {args.transfer_mode}有误')

    elif args.enforce_refinement:

        if args.refinement_mode == "test":
            pass

        elif args.refinement_mode == "dataset":

            
            if args.refinement_inherient_file is None:
                raise ValueError("请提供润色前的标题文件")
            if args.refinement_sample_dataset is None:
                raise ValueError("请提供采样数据文件")
            
            inherient_title = load_json(args.refinement_inherient_file)
            sample_data = load_json(args.refinement_sample_dataset)
            unique_index = 0
            while os.path.exists(os.path.join(args.refinement_save_dir, f"polish_{unique_index}.json")):
                unique_index += 1

            out_path = os.path.join(args.refinement_save_dir, f"polish_{unique_index}.json")
            out_data = {}
            
            for group_name, group_value in tqdm(sample_data.items()):
                group_anonymous_list = [name for name in group_value if name != "count"]
                for anonymous_name in group_anonymous_list:
                    anonymous_data = sample_data[group_name][anonymous_name]
                    
                    history_note_list = anonymous_data["history"] 
                    prefix_query = "某用户的发帖历史标题为："
                    for i, history_note in enumerate(history_note_list):
                        prefix_query = prefix_query + f"第{i+1}条：{history_note['title']}，"
                    prefix_query += f"最新发帖的标题为：{inherient_title[anonymous_name]};"
                    suffix_query = "请你检查历史标题中是否有符号或固定格式，并润色新发帖标题，使其满足用户的符号偏好"
                    query = prefix_query + suffix_query
                    query = "某用户的发帖历史标题为：第1条：公园20分钟效应｜它真的有用！，第2条：下雨了，快跑呀！，最新发帖的标题为：贵州之旅：古朴村落与自然风光；请你检查历史标题中是否有符号或固定格式，并润色新发帖标题，使其满足用户的符号偏好"
                    import pdb; pdb.set_trace()

                    # 构造推理kwargs
                    kwargs = {'query': query}
                    system = template.default_system
                    data = {"system": system}
            
                    # 获取data参数
                    history = data.get('history')
                    system = data.get('system')
                    # import pdb; pdb.set_trace()
                    tools = data.get('tools')
                    objects = data.get('objects')
                    # 如果采用了详尽输出且system不为None，打印system
                    if args.verbose and system is not None:
                        print(f'[SYSTEM]{system}')
                    # ------处理历史对话------
                    if history is None:
                        history = []
                    # 加入kwargs字典
                    kwargs['history'] = history
                    # ------处理system------
                    if system is None and template.use_default_system:
                        system = template.default_system
                    # 加入kwargs字典
                    kwargs['system'] = system
                    # ------处理多媒体文件------
                    # 遍历media_key: [images, audios, videos]
                    for media_key in MediaTag.media_keys.values():
                        media_files = data.get(media_key)
                        # 加入kwargs字典
                        if media_files is not None:
                            kwargs[media_key] = media_files
                    # ------处理tools、objects、truncation_strategy------
                    if tools is not None:
                        kwargs['tools'] = tools
                    if objects is not None:
                        kwargs['objects'] = objects
                    kwargs['truncation_strategy'] = args.truncation_strategy
                    if args.infer_backend in {'vllm', 'lmdeploy'}:
                        assert args.stream
                        if args.verbose:
                            print(f"[QUERY]{data['query']}\n[RESPONSE]", end='')
                        gen = inference_stream_x(llm_engine, template, [kwargs], lora_request=lora_request)
                        print_idx = 0
                        for resp_list in gen:
                            response = resp_list[0]['response']
                            if args.verbose and len(response) > print_idx:
                                print(response[print_idx:], end='', flush=True)
                                print_idx = len(response)
                        print()
                    # 调用inference接口得到response
                    else:
                        response, _ = inference(
                            model, template, stream=args.stream and args.verbose, verbose=args.verbose, **kwargs)
                    # 若data中包含了标签信息，赋值为label
                    label = data.pop('response', None)
                    # 构建对话保存实例
                    obj = {
                        'system': kwargs['system'],
                        'query': kwargs['query'],
                        'response': response,
                        'label': label,
                        'history': kwargs['history'],
                    }
                    # 将多媒体文件路径加入对话保存实例
                    for media_key in MediaTag.media_keys.values():
                        media_files = kwargs.get(media_key)
                        if media_files is not None:
                            obj[media_key] = media_files
                    # 输出到对话存储文件中
                    if jsonl_path is not None:
                        append_to_jsonl(jsonl_path, obj)
                    # 将对话保存实例加入result中
                    result.append(obj)

                    # 将回复加入out_data
                    out_data[anonymous_name] = response

                    if args.verbose:
                        print()
                        print(f'[LABELS]{label}')
                        for media_key in MediaTag.media_keys.values():
                            media_files = kwargs.get(media_key)
                            if media_files is not None:
                                print(f'[{media_key.upper()}]{media_files}')
                        print('-' * 50, flush=True)
                # save
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(out_data, f, ensure_ascii=False, indent=4)


        elif args.refinement_mode == "lora-dataset":

            if args.refinement_inherient_file is None:
                raise ValueError("请提供润色前的标题文件")
            if args.refinement_sample_dataset is None:
                raise ValueError("请提供采样数据文件")
            
            inherient_title = load_json(args.refinement_inherient_file)
            sample_data = load_json(args.refinement_sample_dataset)
            unique_index = 0
            while os.path.exists(os.path.join(args.refinement_save_dir, f"polish_{unique_index}.json")):
                unique_index += 1

            out_path = os.path.join(args.refinement_save_dir, f"polish_{unique_index}.json")
            out_data = {}
            
            for group_name, group_value in tqdm(sample_data.items()):
                group_anonymous_list = [name for name in group_value if name != "count"]
                for anonymous_name in group_anonymous_list:
                    anonymous_data = sample_data[group_name][anonymous_name]
                    
                    prompt = "我正在分析每位博主的历史发帖标题以分析其符号使用的偏好，并对该博主的最新发帖标题作出润色建议，使最新发帖标题的符号使用个性与历史发帖保持一致。我将会依次给你提供每位博主的历史发帖标题和最新发帖标题，请你根据历史发帖分析该博主的符号偏好，并对最新发帖标题给出润色建议。现在，让我们开始吧！"
                    
                    history_note_list = anonymous_data["history"] 
                    prefix_query = "该博主的历史发帖标题为："
                    for i, history_note in enumerate(history_note_list):
                        if i == len(history_note_list) - 1:
                            prefix_query = prefix_query + f"{i+1}. 标题：《{history_note['title']}》。"
                        else:
                            prefix_query = prefix_query + f"{i+1}. 标题：《{history_note['title']}》;"

                    prefix_query += f"该用户的最新发帖标题为：《{inherient_title[anonymous_name]}》"
                    # suffix_query = "请你对当前用户进行分析，并润色最新发帖标题："
                    query = prompt + prefix_query
                    # query = query + prefix_query
                    # import pdb; pdb.set_trace()

                    # 构造推理kwargs
                    kwargs = {'query': query}
                    system = template.default_system
                    data = {"system": system}
            
                    # 获取data参数
                    history = data.get('history')
                    system = data.get('system')
                    # import pdb; pdb.set_trace()
                    tools = data.get('tools')
                    objects = data.get('objects')
                    # 如果采用了详尽输出且system不为None，打印system
                    if args.verbose and system is not None:
                        print(f'[SYSTEM]{system}')
                    # ------处理历史对话------
                    if history is None:
                        history = []
                    # 加入kwargs字典
                    kwargs['history'] = history
                    # ------处理system------
                    if system is None and template.use_default_system:
                        system = template.default_system
                    # 加入kwargs字典
                    kwargs['system'] = system
                    # ------处理多媒体文件------
                    # 遍历media_key: [images, audios, videos]
                    for media_key in MediaTag.media_keys.values():
                        media_files = data.get(media_key)
                        # 加入kwargs字典
                        if media_files is not None:
                            kwargs[media_key] = media_files
                    # ------处理tools、objects、truncation_strategy------
                    if tools is not None:
                        kwargs['tools'] = tools
                    if objects is not None:
                        kwargs['objects'] = objects
                    kwargs['truncation_strategy'] = args.truncation_strategy
                    if args.infer_backend in {'vllm', 'lmdeploy'}:
                        assert args.stream
                        if args.verbose:
                            print(f"[QUERY]{data['query']}\n[RESPONSE]", end='')
                        gen = inference_stream_x(llm_engine, template, [kwargs], lora_request=lora_request)
                        print_idx = 0
                        for resp_list in gen:
                            response = resp_list[0]['response']
                            if args.verbose and len(response) > print_idx:
                                print(response[print_idx:], end='', flush=True)
                                print_idx = len(response)
                        print()
                    # 调用inference接口得到response
                    else:
                        response, _ = inference(
                            model, template, stream=args.stream and args.verbose, verbose=args.verbose, **kwargs)
                    # 若data中包含了标签信息，赋值为label
                    label = data.pop('response', None)
                    # 构建对话保存实例
                    obj = {
                        'system': kwargs['system'],
                        'query': kwargs['query'],
                        'response': response,
                        'label': label,
                        'history': kwargs['history'],
                    }
                    # 将多媒体文件路径加入对话保存实例
                    for media_key in MediaTag.media_keys.values():
                        media_files = kwargs.get(media_key)
                        if media_files is not None:
                            obj[media_key] = media_files
                    # 输出到对话存储文件中
                    if jsonl_path is not None:
                        append_to_jsonl(jsonl_path, obj)
                    # 将对话保存实例加入result中
                    result.append(obj)

                    # 将回复加入out_data
                    out_data[anonymous_name] = response

                    if args.verbose:
                        print()
                        print(f'[LABELS]{label}')
                        for media_key in MediaTag.media_keys.values():
                            media_files = kwargs.get(media_key)
                            if media_files is not None:
                                print(f'[{media_key.upper()}]{media_files}')
                        print('-' * 50, flush=True)
                # save
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(out_data, f, ensure_ascii=False, indent=4)

        else:
            raise ValueError(f"你提供的润色模式：{args.refinement_mode}不符合要求")

    ### 此时表示args.eval_human为False (& args.eval_ficl为False)
    else:
        # 设置数据集参数
        dataset_kwargs = {
            'dataset_seed': args.dataset_seed,
            'check_dataset_strategy': args.check_dataset_strategy,
            'model_name': args.model_name,
            'model_author': args.model_author
        }
        # 获取验证集
            # 如果提供了args.val_dataset，则获取全部验证集(比例为1.0)
            # 如果未提供，则从args.dataset中抽取dataset_test_ratio比例作为验证集
        if len(args.val_dataset) > 0:
            _, val_dataset = get_dataset(args.val_dataset, 1.0, **dataset_kwargs)
        else:
            _, val_dataset = get_dataset(args.dataset, args.dataset_test_ratio, **dataset_kwargs)
        # 处理数据集兼容性
        _, val_dataset = args._handle_dataset_compat(_, val_dataset)
        # 确保此时val_dataset不为None
        assert val_dataset is not None
        # 如果args.show_dataset_sample在验证集有效数量范围内（但一般默认为-1）
            # 则随机采样验证集
        if 0 <= args.show_dataset_sample < val_dataset.shape[0]:
            random_state = np.random.RandomState(args.dataset_seed)
            logger.info(f'show_dataset_sample: {args.show_dataset_sample}')
            val_dataset = sample_dataset(val_dataset, args.show_dataset_sample, random_state)
        # 记录验证集
        logger.info(f'val_dataset: {val_dataset}')
        # 设置详细模式（verbose为True则详尽输出，verbose为False则简洁输出）
        # 如果args.verbose为None，根据采样验证集数量初始化args.verbose
        if args.verbose is None:
            # 如果采样验证集数量>=20，则采用简洁输出
            if len(val_dataset) >= 20:
                args.verbose = False
            # 否则，采用详尽输出
            else:
                args.verbose = True
            logger.info(f'Setting args.verbose: {args.verbose}')
        # 如果同时设置了简洁输出和流式推理，则将流式推理设置为False
        if not args.verbose and args.stream:
            args.stream = False
            logger.info(f'Setting args.stream: {args.stream}')
        ### 如果推理后端在{'vllm', 'lmdeploy'}中，且采用非流式推理
        if args.infer_backend in {'vllm', 'lmdeploy'} and not args.stream:
            # 调整args.verbose为False
            if args.verbose:
                args.verbose = False
                logger.info('Setting args.verbose: False')
            # 处理标签列表
            label_list = None
            if 'response' in val_dataset.features:
                label_list = val_dataset['response']
                val_dataset = val_dataset.remove_columns('response')
            # 准备请求列表
            request_list = []
            # 遍历验证集中的每个数据
            for data in val_dataset:
                request = {'query': data['query']}
                # 获取data history
                history = data.get('history')
                # 获取data system
                system = data.get('system')
                if history is None:
                    history = []
                # 将history和system加入request中
                request['history'] = history
                if system is None and template.use_default_system:
                    system = template.default_system
                request['system'] = system
                # 将媒体文件加入request中
                for media_key in MediaTag.media_keys.values():
                    media_files = data.get(media_key)
                    if media_files is not None:
                        request[media_key] = media_files
                request['truncation_strategy'] = args.truncation_strategy
                request_list.append(request)
            # 调用inference_x获取response_list
            resp_list = inference_x(llm_engine, template, request_list, use_tqdm=True)
            result = []
            # 若label_list不为空，为每个request添加'label'标签
            if label_list is not None:
                for request, label in zip(request_list, label_list):
                    request['label'] = label
            # 遍历每个数据（包含request和模型生成的回复）
                # 创建obj并进行结果写入
            for request, resp in zip(request_list, resp_list):
                obj = {
                    'system': request['system'],
                    'query': request['query'],
                    'response': resp['response'],
                    'label': request.pop('label', None),
                    'history': request['history'],
                }
                for media_key in MediaTag.media_keys.values():
                    media_files = request.get(media_key)
                    if media_files is not None:
                        obj[media_key] = media_files
                # 写入jsonl_path
                if jsonl_path is not None:
                    append_to_jsonl(jsonl_path, obj)
                # 添加至result
                result.append(obj)
        ### 推理后端非'vllm'或'lmdeploy'或args.stream设置为True
        else:
            # 如果未采取详尽输出，则可为val_dataset添加tqdm进度条
            if not args.verbose:
                val_dataset = tqdm(val_dataset)
            # 遍历val_dataset
            for data in val_dataset:
                # 构造kwargs字典用于模型传参
                kwargs = {'query': data['query']}
                # 获取data参数
                history = data.get('history')
                system = data.get('system')
                tools = data.get('tools')
                objects = data.get('objects')
                # 如果采用了详尽输出且system不为None，打印system
                if args.verbose and system is not None:
                    print(f'[SYSTEM]{system}')
                # ------处理历史对话------
                if history is None:
                    history = []
                # 加入kwargs字典
                kwargs['history'] = history
                # ------处理system------
                if system is None and template.use_default_system:
                    system = template.default_system
                # 加入kwargs字典
                kwargs['system'] = system
                # ------处理多媒体文件------
                # 遍历media_key: [images, audios, videos]
                for media_key in MediaTag.media_keys.values():
                    media_files = data.get(media_key)
                    # 加入kwargs字典
                    if media_files is not None:
                        kwargs[media_key] = media_files
                # ------处理tools、objects、truncation_strategy------
                if tools is not None:
                    kwargs['tools'] = tools
                if objects is not None:
                    kwargs['objects'] = objects
                kwargs['truncation_strategy'] = args.truncation_strategy
                if args.infer_backend in {'vllm', 'lmdeploy'}:
                    assert args.stream
                    if args.verbose:
                        print(f"[QUERY]{data['query']}\n[RESPONSE]", end='')
                    gen = inference_stream_x(llm_engine, template, [kwargs], lora_request=lora_request)
                    print_idx = 0
                    for resp_list in gen:
                        response = resp_list[0]['response']
                        if args.verbose and len(response) > print_idx:
                            print(response[print_idx:], end='', flush=True)
                            print_idx = len(response)
                    print()
                # 调用inference接口得到response
                else:
                    response, _ = inference(
                        model, template, stream=args.stream and args.verbose, verbose=args.verbose, **kwargs)
                # 若data中包含了标签信息，赋值为label
                label = data.pop('response', None)
                # 构建对话保存实例
                obj = {
                    'system': kwargs['system'],
                    'query': kwargs['query'],
                    'response': response,
                    'label': label,
                    'history': kwargs['history'],
                }
                # 将多媒体文件路径加入对话保存实例
                for media_key in MediaTag.media_keys.values():
                    media_files = kwargs.get(media_key)
                    if media_files is not None:
                        obj[media_key] = media_files
                # 输出到对话存储文件中
                if jsonl_path is not None:
                    append_to_jsonl(jsonl_path, obj)
                # 将对话保存实例加入result中
                result.append(obj)
                if args.verbose:
                    print()
                    print(f'[LABELS]{label}')
                    for media_key in MediaTag.media_keys.values():
                        media_files = kwargs.get(media_key)
                        if media_files is not None:
                            print(f'[{media_key.upper()}]{media_files}')
                    print('-' * 50, flush=True)

    if jsonl_path is not None:
        logger.info(f'save_result_path: {jsonl_path}')
    return {'result': result}


infer_main = get_main(InferArguments, llm_infer)
merge_lora_main = get_main(InferArguments, merge_lora)


