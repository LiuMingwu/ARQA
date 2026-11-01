import os
import json
import re
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

from retrival_clean import Retrieval_BGE

os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "EMPTY"

LLM_MODEL = "qwen3"
LLM_PROVIDER = "vllm"
ENABLE_THINKING = False
BGE_MODEL_PATH = "/path/to/bge-m3"

client = OpenAI()


def retrieve(d, model_name, question_text_list, triple_list, n=25):
    retriever = Retrieval_BGE(d, model_name, question_text_list, triple_list)
    distances, corpus_ids = retriever.compute_similarity(n=n)
    return retriever.get_result(distances, corpus_ids, question_text_list, re_rank=False)


REWRITE_PROMPT_BASE = """You are given a question with an implicit event/time reference, and a list of historical facts.

Step 1 — Select: Scan the facts and pick the one whose event matches the implicit reference in the question.
Step 2 — Rewrite: Replace the implicit reference with the explicit time from the selected fact.

Rules:
1. Only change the implicit event/time phrase. Keep entity names and all other wording exactly unchanged.
2. If no fact matches, set "selected_fact" to null, "rewritten_question" to the original.
3. Output ONLY a JSON object, no other text.

{examples}

Now your turn:
Facts:
{facts}

Original: {question}

Output:"""

REWRITE_EXAMPLES_MULTITQ = """Example 1 (after_first):
Facts:
- Denmark Ministry of Defence Make a visit Iraq in 2006-01-05
- Iran Host a visit Iraq in 2006-01-04
Original: After the Danish Ministry of Defence and Security, who was the first to visit Iraq?
{"selected_fact": "Denmark Ministry of Defence Make a visit Iraq in 2006-01-05", "rewritten_question": "After 2006-01-05, who was the first to visit Iraq?"}

Example 2 (before_last):
Facts:
- China Make an appeal to Taiwan in 2007-10-31
- Taiwan Ministry of Defence Make a request to China in 2007-09-15
Original: Before the military of Taiwan, which country did China threaten last?
{"selected_fact": "Taiwan Ministry of Defence Make a request to China in 2007-09-15", "rewritten_question": "Before 2007-09-15, which country did China threaten last?"}

Example 3 (equal_multi):
Facts:
- UAE Announce diplomatic cooperation in 2015-07-17
- Kuwait Announce diplomatic cooperation in 2015-07-17
Original: On the same day as the United Arab Emirates, Kuwait announced its intention to join diplomatic cooperation with which country?
{"selected_fact": "UAE Announce diplomatic cooperation in 2015-07-17", "rewritten_question": "On 2015-07-17, Kuwait announced its intention to join diplomatic cooperation with which country?"}

Example 4 (no match):
Facts:
- Iraq Make a visit Iran in 2006-01-04
Original: In which year did Taiwan last make a request to China?
{"selected_fact": null, "rewritten_question": "In which year did Taiwan last make a request to China?"}"""

REWRITE_EXAMPLES_CRON = """Example 1 (time_join):
Facts:
- Patrick Doherty position held Member of the 1st Northern Ireland Assembly from 1998 to 2003
- John Smith position held Minister of Education from 2001 to 2005
Original: When Patrick Doherty was the Member of the 1st Northern Ireland Assembly, who was Member of the 1st Northern Ireland Assembly?
{"selected_fact": "Patrick Doherty position held Member of the 1st Northern Ireland Assembly from 1998 to 2003", "rewritten_question": "During 1998 to 2003, who was Member of the 1st Northern Ireland Assembly?"}

Example 2 (before_after):
Facts:
- Thomas Jefferson position held Vice President of the United States from 1797 to 1801
- John Adams position held President of the United States from 1797 to 1801
Original: Who held the position of President after Thomas Jefferson?
{"selected_fact": "Thomas Jefferson position held Vice President of the United States from 1797 to 1801", "rewritten_question": "Who held the position of President after 1801?"}

Example 3 (no match):
Facts:
- Richard Richards position held Member of the 15th Parliament of the United Kingdom from 1847 to 1852
Original: What year was the first Nobel Prize awarded?
{"selected_fact": null, "rewritten_question": "What year was the first Nobel Prize awarded?"}"""

def strip_thinking(text):
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def extract_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def gpt_chat_completion(messages, max_retries=5):
    backoff_time = 1
    for attempt in range(max_retries):
        try:
            kwargs = {
                "messages": messages,
                "model": LLM_MODEL,
                "temperature": 0.7,
            }
            if LLM_PROVIDER == "deepseek":
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            elif LLM_PROVIDER == "vllm":
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}
                }
            response = client.chat.completions.create(**kwargs)
            return strip_thinking(response.choices[0].message.content or "")
        except Exception as e:
            print(f"[Attempt {attempt + 1}/{max_retries}] Error: {e}")
            if hasattr(e, "status_code") and 400 <= e.status_code < 500:
                raise
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff_time)
            backoff_time *= 1.5


def rewrite(fact_list, question_list, dataset, checkpoint_path=None,
            max_workers=8):
    n = len(question_list)
    results = {}
    lock = Lock()

    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        raw = checkpoint.get("results", {})
        if isinstance(raw, list):
            results = {
                i: {"rewritten_question": v, "selected_fact": None}
                for i, v in enumerate(raw)
            }
        else:
            results = {
                int(k): (
                    v
                    if isinstance(v, dict)
                    else {"rewritten_question": v, "selected_fact": None}
                )
                for k, v in raw.items()
            }

    rewrite_tasks = []
    for i in range(n):
        if i in results:
            continue
        qtype = question_list[i].get("qtype", "")
        if qtype in (
            "before_after",
            "after_first",
            "before_last",
            "equal_multi",
            "time_join",
        ):
            rewrite_tasks.append(
                (i, question_list[i]["question"], qtype, fact_list[i]["fact"])
            )
        else:
            results[i] = {
                "rewritten_question": question_list[i]["question"],
                "selected_fact": None,
            }

    def save_checkpoint():
        if checkpoint_path:
            with lock:
                str_results = {str(k): v for k, v in results.items()}
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"results": str_results, "count": len(results)},
                        f,
                        ensure_ascii=False,
                    )

    def rewrite_one(idx, question, qtype, facts):
        facts_text = "\n".join(f"- {f}" for f in facts[:5])
        examples = (
            REWRITE_EXAMPLES_CRON
            if dataset == "CronQuestions"
            else REWRITE_EXAMPLES_MULTITQ
        )
        prompt = REWRITE_PROMPT_BASE.replace("{examples}", examples)
        prompt = prompt.replace("{facts}", facts_text).replace("{question}", question)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = gpt_chat_completion(messages=messages, max_retries=3)
        except Exception:
            return idx, {"rewritten_question": question, "selected_fact": None}

        result = extract_json(response)
        candidate = result.get("rewritten_question", question)
        selected_fact = result.get("selected_fact")
        if selected_fact is not None and not isinstance(selected_fact, str):
            selected_fact = str(selected_fact)

        return idx, {
            "rewritten_question": candidate,
            "selected_fact": selected_fact,
        }

    completed_since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(rewrite_one, idx, q, qt, facts): idx
            for idx, q, qt, facts in rewrite_tasks
        }
        for future in tqdm(
            as_completed(futures), total=len(rewrite_tasks), desc="rewrite"
        ):
            idx, result = future.result()
            results[idx] = result
            completed_since_checkpoint += 1
            if checkpoint_path and completed_since_checkpoint % 50 == 0:
                save_checkpoint()

    if checkpoint_path:
        save_checkpoint()

    new_questions = [results[i]["rewritten_question"] for i in range(n)]

    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return new_questions


def load_triples(path):
    triple_list = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            triple_list.append(line.strip().replace("_", " ").split("\t"))
    return triple_list


def load_questions(path):
    with open(path, "r") as file:
        question_json = json.load(file)

    question_list = []
    question_text_list = []
    valid_indices = []

    for i, sample in enumerate(question_json):
        question = sample.get("question", "")
        if not question:
            input_text = sample.get("input", "")
            match = re.search(r"Question:\s*\{(.*?)\}", input_text, flags=re.DOTALL)
            question = match.group(1).strip() if match else ""
            question = re.sub(r"^Answer:\s*", "", question, flags=re.IGNORECASE).strip()
        if not question:
            continue

        valid_indices.append(i)
        question_list.append(
            {
                "question": question,
                "qtype": sample.get("qtype", ""),
            }
        )
        question_text_list.append(question)

    return question_json, question_list, question_text_list, valid_indices


def main():
    triple_list = load_triples(args.triplet_path)
    question_json, question_list, question_text_list, valid_indices = load_questions(
        args.question_path
    )

    background_list = retrieve(
        args.d, BGE_MODEL_PATH, question_text_list, triple_list, n=args.n_background
    )

    rewrite_questions = rewrite(
        background_list,
        question_list,
        dataset=args.d,
        checkpoint_path=f"{args.rewrite_output_path}.checkpoint.json",
        max_workers=args.max_workers,
    )

    fact_list_100 = retrieve(
        args.d, BGE_MODEL_PATH, rewrite_questions, triple_list, n=100
    )

    subgraph_path = args.rewrite_output_path.replace(".json", "_subgraph.json")
    with open(subgraph_path, "w", encoding="utf-8") as f:
        json.dump(fact_list_100, f, ensure_ascii=False, indent=2)

    new_data = []
    for j, orig_idx in enumerate(valid_indices):
        sample = question_json[orig_idx]
        new_sample = sample.copy()
        new_question = rewrite_questions[j]
        facts = fact_list_100[j]["fact"][: args.n_facts]
        new_sample["question"] = new_question
        new_sample["input"] = (
            f"Historical facts:{facts}\nQuestion:{{{new_question}}}"
        )
        new_data.append(new_sample)

    with open(args.rewrite_output_path, "w", encoding="utf-8") as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--triplet_path", type=str)
    argparser.add_argument("--question_path", type=str)
    argparser.add_argument(
        "--rewrite_output_path", type=str, required=True,
        help="Path for the rewritten dataset JSON",
    )
    argparser.add_argument("--d", default="CronQuestions")
    argparser.add_argument(
        "--n_background", type=int, default=5,
        help="Number of facts for first-stage retrieval (background)",
    )
    argparser.add_argument(
        "--n_facts", type=int, default=50,
        help="Number of facts for second-stage retrieval (after rewrite)",
    )
    argparser.add_argument(
        "--max_workers", type=int, default=8,
        help="Number of concurrent workers for LLM inference",
    )

    args = argparser.parse_args()
    main()
