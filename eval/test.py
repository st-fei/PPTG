
import os
import json
from tools import load_json
from PIL import Image
from cal_metrics import cal_nlgeval_metrics

sample_file = "/home/tfshen/pyproject/pcg/data/summary4eval/sample.json"
save_file = "/home/tfshen/pyproject/pcg/data/summary4eval/group_index.json"

sample_data = load_json(sample_file)
group_index = {}
for group_name, group_value in sample_data.items():
    group_anonymous_list = [name for name in group_value if name != 'count']
    for anonymous_name in group_anonymous_list:
        group_index[anonymous_name] = group_name

with open(save_file, 'w', encoding='utf-8') as f:
    json.dump(group_index, f, ensure_ascii=False, indent=4)