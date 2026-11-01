import os
import json
import re
import string
import ast
from datetime import datetime

from tqdm import tqdm
from llmtuner import ChatModel
from llmtuner.tuner.core import get_infer_args
from pro_model.totoken import data_load_retrieval


def normalize(s: str) -> str:
    s = str(s)
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_hit(prediction, answer):
    answer = ast.literal_eval(answer)
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def extract_topk_prediction(prediction, k=1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]


def normalize_qtype(qt: str) -> str:
    if qt is None:
        return 'unknown'
    s = str(qt).strip().lower()
    s = s.replace(' ', '_').replace('/', '_').replace('-', '_')
    if s in ('beforeafter', 'before_after', 'before/after'):
        return 'before_after'
    if s in ('equalmulti', 'equal_multi', 'equal multi'):
        return 'equal_multi'
    if s.startswith('equal'):
        return 'equal'
    return s


def list_k(str_list, k=10):
    content = str_list.strip("[]")
    elements = content.split(",")

    if elements and elements[0]:
        selected_elements = []
        for i in range(min(k, len(elements))):
            element = elements[i].strip().strip("'\"")
            selected_elements.append(f"'{element}'")
        return f"[{', '.join(selected_elements)}]"
    return str([])


def eval_result(predict_file, cal_f1=True, topk=1):
    eval_name = f"detailed_eval_result_top_{topk}.jsonl" if topk > 0 else 'detailed_eval_result.jsonl'
    detailed_eval_file = predict_file.replace('predictions.jsonl', eval_name)

    hit_list = []
    q_counts = {}
    q_hits = {}
    qlabel_counts = {}
    qlabel_hits = {}
    ans_counts = {}
    ans_hits = {}
    time_levels_stats = {}
    target_time_levels = ['day', 'month', 'year']
    target_qtypes = ['equal', 'before_after', 'equal_multi']
    for tl in target_time_levels:
        time_levels_stats[tl] = {qt: {'count': 0, 'hits': 0} for qt in target_qtypes}

    with open(predict_file, 'r') as f, open(detailed_eval_file, 'w') as f2:
        for line in f:
            try:
                data = json.loads(line)
            except Exception:
                print(line)
                continue
            prediction = data.get('prediction')
            prediction = list_k(prediction, k=topk)

            answer = data.get('ground_truth')
            qtype_raw = data.get('qtype', data.get('question_type', data.get('q_type', 'unknown')))
            qtype = normalize_qtype(qtype_raw)
            qlabel = data.get('qlabel', data.get('q_label', 'unknown'))
            answer_type = data.get('answer_type', data.get('ans_type', 'unknown'))
            time_level = str(data.get('time_level', 'unknown')).lower()

            if cal_f1:
                if not isinstance(prediction, list):
                    prediction = prediction.split("\n") if isinstance(prediction, str) else [str(prediction)]
                else:
                    prediction = extract_topk_prediction(prediction, topk)

                prediction_str = ' '.join(prediction)
                hit = eval_hit(prediction_str, answer)
                hit_list.append(hit)
                f2.write(json.dumps({
                    'prediction': prediction,
                    'ground_truth': answer,
                    'qtype': qtype,
                    'qtype_raw': qtype_raw,
                    'qlabel': qlabel,
                    'answer_type': answer_type,
                    'time_level': time_level,
                    'hit': hit,
                }, ensure_ascii=False) + '\n')
            else:
                pred_for_eval = prediction if isinstance(prediction, str) else str(prediction)
                hit = eval_hit(pred_for_eval, answer)
                hit_list.append(hit)
                f2.write(json.dumps({
                    'prediction': prediction,
                    'ground_truth': answer,
                    'qtype': qtype,
                    'qtype_raw': qtype_raw,
                    'qlabel': qlabel,
                    'answer_type': answer_type,
                    'time_level': time_level,
                    'hit': hit,
                }, ensure_ascii=False) + '\n')

            q_counts[qtype] = q_counts.get(qtype, 0) + 1
            q_hits[qtype] = q_hits.get(qtype, 0) + (1 if hit else 0)
            qlabel_counts[qlabel] = qlabel_counts.get(qlabel, 0) + 1
            qlabel_hits[qlabel] = qlabel_hits.get(qlabel, 0) + (1 if hit else 0)
            ans_counts[answer_type] = ans_counts.get(answer_type, 0) + 1
            ans_hits[answer_type] = ans_hits.get(answer_type, 0) + (1 if hit else 0)
            if time_level in target_time_levels and qtype in target_qtypes:
                time_levels_stats[time_level][qtype]['count'] += 1
                time_levels_stats[time_level][qtype]['hits'] += (1 if hit else 0)

    overall_hit = (sum(hit_list) * 100 / len(hit_list)) if len(hit_list) > 0 else 0.0
    print(f"Hit: {overall_hit:.2f}")
    result_name = f"eval_result_top_{topk}.txt" if topk > 0 else 'eval_result.txt'
    eval_result_path = predict_file.replace('predictions.jsonl', result_name)

    has_valid_qtype = any(k != 'unknown' for k in q_counts.keys())
    has_valid_ans_type = any(k != 'unknown' for k in ans_counts.keys())
    has_time_level_stats = any(
        time_levels_stats[tl][qt]['count'] > 0
        for tl in target_time_levels for qt in target_qtypes
    )

    with open(eval_result_path, 'w') as f:
        f.write(f"Overall Hit: {overall_hit:.2f}%\n")

        f.write("\nPer-qlabel:\n")
        for ql in sorted(qlabel_counts.keys()):
            cnt = qlabel_counts[ql]
            hits = qlabel_hits.get(ql, 0)
            pct = hits * 100.0 / cnt if cnt > 0 else 0.0
            f.write(f"{ql}: {hits}/{cnt} ({pct:.2f}%)\n")
            print(f"{ql}: {hits}/{cnt} ({pct:.2f}%)")

        if has_valid_qtype:
            f.write("\nPer-question-type:\n")
            for qt in sorted(q_counts.keys()):
                cnt = q_counts[qt]
                hits = q_hits.get(qt, 0)
                pct = hits * 100.0 / cnt if cnt > 0 else 0.0
                f.write(f"{qt}: {hits}/{cnt} ({pct:.2f}%)\n")
                print(f"{qt}: {hits}/{cnt} ({pct:.2f}%)")

        if has_valid_ans_type:
            f.write("\nPer-answer_type:\n")
            for at in sorted(ans_counts.keys()):
                cnt = ans_counts[at]
                hits = ans_hits.get(at, 0)
                pct = hits * 100.0 / cnt if cnt > 0 else 0.0
                f.write(f"{at}: {hits}/{cnt} ({pct:.2f}%)\n")
                print(f"{at}: {hits}/{cnt} ({pct:.2f}%)")

        if has_time_level_stats:
            f.write("\nTime-level x qtype (targets: equal, before_after, equal_multi):\n")
            for tl in target_time_levels:
                f.write(f"\nTime level: {tl}\n")
                for qt in target_qtypes:
                    cnt = time_levels_stats[tl][qt]['count']
                    hits = time_levels_stats[tl][qt]['hits']
                    pct = hits * 100.0 / cnt if cnt > 0 else 0.0
                    f.write(f"  {qt}: {hits}/{cnt} ({pct:.2f}%)\n")
                    print(f"  {qt}: {hits}/{cnt} ({pct:.2f}%)")


def main():
    model_args, data_args, _, _ = get_infer_args()
    id2sub = data_load_retrieval(data_args)
    chat_model = ChatModel()

    if data_args.input_file:
        input_file = data_args.input_file
    else:
        input_file = os.path.join(data_args.dataset_dir, data_args.dataset, "examples.json")

    if data_args.input_file:
        dataset_name = os.path.basename(os.path.dirname(os.path.abspath(input_file)))
    else:
        dataset_name = data_args.dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("predictions", f"{dataset_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Input:  {input_file}")
    print(f"Output: {output_dir}")

    output_file = os.path.join(output_dir, "predictions.jsonl")

    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        json_data = json.load(f)

        total_lines = 0
        for data in tqdm(json_data):
            total_lines += 1
            query = data['instruction'] + data['input']
            answer = data['output']
            id = data.get('ID', data.get('id', None))
            subgraph = id2sub[id]
            qlabel = data.get('qlabel', data.get('q_label', None))

            predict = chat_model.chat(query, subgraph)[0]

            answer_type = data.get('answer_type', data.get('ans_type', None))
            qtype = data.get('qtype', data.get('question_type', data.get('q_type', None)))
            time_level = data.get('time_level', None)

            result = {
                "ID": id,
                "input": query,
                "ground_truth": answer,
                "prediction": predict,
            }
            if qlabel is not None:
                result['qlabel'] = qlabel
            if answer_type is not None:
                result['answer_type'] = answer_type
            if qtype is not None:
                result['qtype'] = qtype
            if time_level is not None:
                result['time_level'] = time_level
            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
            out_f.flush()
            if total_lines % 2000 == 0:
                for k in (1, 10):
                    eval_result(output_file, topk=k)

    for k in (1, 10):
        eval_result(output_file, topk=k)


if __name__ == "__main__":
    main()
