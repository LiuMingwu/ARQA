from typing import List
import json
import math
import re


def data_load_retrieval(data_args):
    if 'MultiTQ' in data_args.dataset:
        dataset = 'MultiTQ'
    else:
        dataset = 'CronQuestions'
    splits = ['train', 'test']

    subgraphs = []
    for split in splits:
        with open(f'/ARQA/data/{dataset}_{split}_subgraphs_bge.json', 'r') as f:
            subgraphs.extend(json.load(f))
    id2subgraph = {}
    for subgraph in subgraphs:
        id2subgraph[subgraph['ID']] = subgraph['fact'][:data_args.topk]
    return id2subgraph


def get_ids(tokenizer, sequence, max_length):
    sequence = tokenizer.encode(sequence)
    sequence = [tokenizer.bos_token_id] + sequence + [tokenizer.eos_token_id]
    if len(sequence) < max_length:
        sequence = sequence + [tokenizer.pad_token_id] * (max_length - len(sequence))
    return sequence[:max_length]


def get_extra_input_ids(tokenizer, subgraphs: List[str], max_length: int):
    subgraph_ids = []
    for subgraph in subgraphs:
        subgraph_ids.append(get_ids(tokenizer, subgraph, max_length))
    return subgraph_ids


def build_gate_ids(tokenizer, query: str, gate_len: int):
    matches = re.findall(r'\{(.*?)\}', query)
    content = matches[0] if matches else query
    gate_ids = tokenizer.encode(content)
    if len(gate_ids) > gate_len - 1:
        gate_ids = gate_ids[:gate_len - 1]
    pad_len = gate_len - len(gate_ids) - 1
    return gate_ids + [tokenizer.pad_token_id] * pad_len + [tokenizer.eos_token_id]
