#!/usr/bin/env python3
r"""Controlled birthplace-question ablation for an existing CoVe list-task pilot.

Reuses the EXACT baseline responses from an already completed community CoVe
batch run. Changes only verification QUESTION FORMAT from yes/no to independent
open-ended birthplace queries ("Where was X born?"). It then uses the original
community final-revision prompt, with the same model and sampling settings.

This is an exploratory question-format ablation, NOT an exact replication of the
original paper. Generated answers need human auditing before Precision is reported.

Requires: openai, python-dotenv; OPENAI_API_KEY set locally or in src/.env or ../.env.
Run from src (PowerShell):
  python .\run_open_birthplace_ablation.py `
    --input .\chicago_cove_pilot_3.jsonl `
    --output .\chicago_open_birthplace_3.jsonl
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / '.env')
load_dotenv(ROOT / '.env')

# Preserve the source repository's final refinement prompt verbatim.
from prompts import FINAL_REFINED_PROMPT, EXECUTE_PLAN_PROMPT_SELF_LLM


def norm(s):
    return re.sub(r'\s+', ' ', str(s).casefold().strip())


def extract_numbered_names(text):
    marker = re.search(r'final\s+refined\s+answer\s*:', text, flags=re.I)
    if marker:
        text = text[marker.end():]
    names, seen = [], set()
    for line in text.splitlines():
        match = re.match(r'^\s*(?:\d+[.)]|[-*•])\s+(.+?)\s*$', line)
        if not match:
            continue
        candidate = match.group(1).strip(' `*_.;:,')
        candidate = re.split(r'\s+(?:[-–—]\s+|\([^)]{20,}\))', candidate, maxsplit=1)[0].strip()
        key = norm(candidate)
        if candidate and len(candidate) < 100 and key not in seen:
            seen.add(key)
            names.append(candidate)
    return names


def extract_last_final_numbered_names(text):
    """Only parse the LAST explicit final answer block (never the excluded list)."""
    markers = list(re.finditer(
        r"final\s+refined\s+answer(?:\s+is)?\s*:", text, flags=re.I
    ))
    if not markers:
        raise ValueError("No explicit Final Refined Answer marker; review raw model output")
    tail = text[markers[-1].end():]
    names = []
    started = False
    for line in tail.splitlines():
        m = re.match(r"^\s*\d+[.)]\s+(.+?)\s*$", line)
        if m:
            started = True
            candidate = m.group(1).strip(" `*_.;:, ")
            candidate = re.split(
                r"\s+[-–—]\s+|\s+\(born\b", candidate, maxsplit=1, flags=re.I
            )[0].strip()
            names.append(candidate)
        elif started and line.strip():
            break
    if not names or len({norm(x) for x in names}) != len(names):
        raise ValueError("Final list missing or contains duplicate names; review raw model output")
    return names


def single_completion(client, model, temperature, max_tokens, text):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{'role': 'user', 'content': text}],
    )
    content = response.choices[0].message.content
    if content is None or not content.strip():
        raise RuntimeError('Received an empty LLM response')
    return content.strip(), {
        'response_id': response.id,
        'model_returned': response.model,
        'prompt_tokens': response.usage.prompt_tokens if response.usage else None,
        'completion_tokens': response.usage.completion_tokens if response.usage else None,
    }


def run_record(client, r, model, temperature, max_tokens):
    if r.get('status') != 'ok' or r.get('retrieval_enabled') is not False:
        raise ValueError('Input requires a successful, closed-book original batch record')
    chain = r['raw_chain_result']
    question = r['question']
    baseline_text = chain['baseline_response']
    names = [x['name'] for x in r['baseline']['entities']]
    if not names:
        raise ValueError('No baseline names found; manual review required')
    if extract_numbered_names(baseline_text) != names:
        raise ValueError('Baseline name parser mismatch: inspect the original batch log')
    qa, calls = [], []
    for name in names:
        verification_question = f'Where was {name} born?'
        # Independent model call: no baseline, original city or previous Q&A.
        prompt = EXECUTE_PLAN_PROMPT_SELF_LLM.format(verification_question=verification_question)
        answer, meta = single_completion(client, model, temperature, max_tokens, prompt)
        qa.append({'entity': name, 'question': verification_question, 'answer': answer})
        calls.append({'stage': 'independent_verification', 'entity': name, **meta})
        print(f'    VERIFY {name}: {answer[:100]}', flush=True)
        time.sleep(0.2)
    verification_text = ''.join(
        f"Question: {it['question']} Answer: {it['answer']}\n" for it in qa
    )
    final_prompt = FINAL_REFINED_PROMPT.format(
        original_question=question,
        baseline_response=baseline_text,
        verification_answers=verification_text,
    )
    final_text, meta = single_completion(client, model, temperature, max_tokens, final_prompt)
    calls.append({'stage': 'final_revision', **meta})
    final_names = extract_last_final_numbered_names(final_text)
    baseline_set = {norm(n) for n in names}
    added = [n for n in final_names if norm(n) not in baseline_set]
    if not final_names:
        print('    WARNING: no final numbered names parsed; inspect final raw text', flush=True)
    if added:
        print(f'    WARNING: final answer added names not in baseline: {added}', flush=True)
    return {
        'status': 'ok',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'question_id': r['question_id'],
        'question': question,
        'method': 'controlled_ablation_open_birthplace_fixed_baseline',
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'retrieval_enabled': False,
        'baseline_source_file_method': r.get('method'),
        'baseline_source_timestamp_utc': r.get('timestamp_utc'),
        'baseline_response': baseline_text,
        'baseline_names': names,
        'verification_questions_answers': qa,
        'final_answer': final_text,
        'final_names': final_names,
        'added_names_needing_review': added,
        'api_calls': calls,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='chicago_cove_pilot_3.jsonl')
    p.add_argument('--output', default='chicago_open_birthplace_3.jsonl')
    p.add_argument('--model', default='gpt-4o-mini')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--max-tokens', type=int, default=1000)
    p.add_argument('--limit', type=int, default=3)
    args = p.parse_args()
    if not os.getenv('OPENAI_API_KEY'):
        p.error('OPENAI_API_KEY missing; set locally or in .env (never upload it)')
    if args.limit < 1:
        p.error('--limit must be at least 1')
    src, dest = Path(args.input), Path(args.output)
    if not src.is_file():
        p.error(f'Input not found: {src.resolve()}')
    if dest.exists():
        p.error(f'Output exists: {dest.resolve()} — choose a fresh output name')
    records = [json.loads(s) for s in src.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    records = records[:args.limit]
    if not records:
        p.error('No input records')
    cli = OpenAI(timeout=90, max_retries=2)
    with dest.open('w', encoding='utf-8') as out:
        for i, record in enumerate(records, 1):
            print(f'[{i}/{len(records)}] {record.get("question_id")}: {record.get("question")}', flush=True)
            try:
                result = run_record(cli, record, args.model, args.temperature, args.max_tokens)
                print(f'    FINAL: {result["final_names"]}', flush=True)
            except Exception as exc:
                result = {
                    'status': 'error', 'question_id': record.get('question_id'),
                    'error_type': type(exc).__name__, 'error_message': str(exc),
                }
                print(f'    ERROR: {exc}', file=sys.stderr, flush=True)
            out.write(json.dumps(result, ensure_ascii=False) + '\n')
            out.flush()
    print(f'Saved {len(records)} records to {dest.resolve()}')
    print('Human audit required for all final names, including any names added by revision.')


if __name__ == '__main__':
    main()
