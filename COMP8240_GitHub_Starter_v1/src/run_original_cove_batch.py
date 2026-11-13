#!/usr/bin/env python3
r"""Batch-run the community Wikidata CoVe chain without external retrieval.

This is intended to be copied into chain-of-verification/src alongside
cove_chains.py, prompts.py, and execute_verification_chain.py.

The summary counts are ONLY dataset-membership matches (gold / near-miss /
unmatched), not final factual precision. Manually audit unmatched names and
incomplete / truncated Wikidata results before reporting metrics.

Example (PowerShell, from src):
  python .\run_original_cove_batch.py --dataset .\near_miss_questions.jsonl `
    --limit 3 --output .\original_cove_batch_3.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / '.env')
load_dotenv(ROOT / '.env')

from langchain_openai import ChatOpenAI
from cove_chains import WikiDataCategoryListCOVEChain
from execute_verification_chain import ExecuteVerificationChain


def normalize(name):
    text = unicodedata.normalize('NFKC', name)
    text = text.replace('’', "'").lower().strip()
    return re.sub(r'\s+', ' ', text).strip(" \t\r\n.;:!?'\"*")


def extract_numbered_names(answer):
    """Simple parser for numbered or bullet lists; review any failures by hand."""
    # The legacy final prompt sometimes outputs TWO duplicate lists, one after
    # 'Final Refined Answer'. Prefer the final designated list in that case.
    marker = re.search(r'final\s+refined\s+answer\s*:', answer, flags=re.I)
    if marker:
        answer = answer[marker.end():]

    names = []
    for line in answer.splitlines():
        m = re.match(r'^\s*(?:\d+[.)]|[-*•])\s+(.+?)\s*$', line)
        if not m:
            continue
        value = re.sub(r'^\*\*(.*?)\*\*$', r'\1', m.group(1).strip())
        value = value.strip('*`_ ').rstrip(' .;,')
        # Do not silently count commentary as names.
        value = re.split(r'\s+(?:[-–—]:?\s+|\([^)]{20,}\))', value, maxsplit=1)[0]
        if value and len(value) < 100:
            names.append(value)

    unique = []
    seen = set()
    for name in names:
        key = normalize(name)
        if key not in seen:
            seen.add(key)
            unique.append(name)
    return unique


def label_names(names, item):
    gold = {normalize(x['name']) for x in item.get('gold', [])}
    near = {normalize(x['name']) for x in item.get('distractors', [])}
    labeled = []
    for name in names:
        key = normalize(name)
        label = 'gold' if key in gold else 'near_miss' if key in near else 'unmatched'
        labeled.append({'name': name, 'dataset_label': label})
    counts = Counter(x['dataset_label'] for x in labeled)
    return {
        'entities': labeled,
        'counts': {
            'gold': counts['gold'],
            'near_miss': counts['near_miss'],
            'unmatched': counts['unmatched'],
            'total': len(labeled),
        },
        'needs_review': bool(counts['unmatched'] or not labeled),
    }


def main():
    p = argparse.ArgumentParser(description='Batch closed-book original-repository Wikidata CoVe reproduction')
    p.add_argument('--dataset', default='near_miss_questions.jsonl')
    p.add_argument('--output', default='original_cove_batch_3.jsonl')
    p.add_argument('--limit', type=int, default=3, help='Number of distinct question records; 0 = all')
    p.add_argument('--id', default='', help='Optional exact question id (e.g. boston-actors)')
    p.add_argument('--model', default='gpt-4o-mini')
    p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--max-tokens', type=int, default=1000)
    p.add_argument('--pause', type=float, default=1.0, help='Seconds between questions')
    args = p.parse_args()

    if not os.getenv('OPENAI_API_KEY'):
        p.error('OPENAI_API_KEY missing. Set it locally or in a .env file; never commit secrets.')
    if args.limit < 0:
        p.error('--limit must be >= 0')

    ds = Path(args.dataset)
    if not ds.exists():
        p.error(f'Dataset not found: {ds.resolve()}')
    with ds.open('r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.id:
        rows = [r for r in rows if r.get('id') == args.id]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        p.error('No questions selected. Check --id and the JSONL data.')

    output = Path(args.output)
    if output.exists():
        p.error(f'Output already exists: {output}. Choose a new filename to prevent overwriting.')

    llm = ChatOpenAI(model_name=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    pipeline = WikiDataCategoryListCOVEChain(llm)()
    verification_steps = 0
    for step in pipeline.chains:
        if isinstance(step, ExecuteVerificationChain):
            step.use_search_tool = False
            step.search_tool = None
            verification_steps += 1
    if verification_steps != 1:
        raise RuntimeError(f'Expected exactly one independent verification chain, found {verification_steps}.')

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as out:
        for index, item in enumerate(rows, start=1):
            qid = item.get('id', str(index))
            question = item['question']
            print(f'[{index}/{len(rows)}] {qid}: {question}', flush=True)
            start = time.perf_counter()
            try:
                result = pipeline.invoke({'original_question': question})
                baseline_names = extract_numbered_names(result.get('baseline_response', ''))
                final_names = extract_numbered_names(result.get('final_answer', ''))
                baseline = label_names(baseline_names, item)
                final = label_names(final_names, item)
                record = {
                    'status': 'ok',
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    'question_id': qid,
                    'question': question,
                    'model': args.model,
                    'temperature': args.temperature,
                    'max_tokens': args.max_tokens,
                    'method': 'community_original_closed_book',
                    'retrieval_enabled': False,
                    'source_dataset': str(ds.name),
                    'source_retrieved_date': item.get('retrieved'),
                    'baseline': baseline,
                    'cove': final,
                    'raw_chain_result': result,
                    'elapsed_sec': round(time.perf_counter() - start, 3),
                }
                print(f"  baseline: {baseline['counts']}")
                print(f"  CoVe:     {final['counts']}")
                if baseline['needs_review'] or final['needs_review']:
                    print('  MANUAL REVIEW REQUIRED: unmatched entries or unparseable list.')
            except Exception as exc:
                record = {
                    'status': 'error',
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    'question_id': qid,
                    'question': question,
                    'model': args.model,
                    'method': 'community_original_closed_book',
                    'error_type': type(exc).__name__,
                    'error_message': str(exc),
                    'elapsed_sec': round(time.perf_counter() - start, 3),
                }
                print(f'  ERROR {record["error_type"]}: {record["error_message"]}', file=sys.stderr)
            out.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
            out.flush()
            if index < len(rows) and args.pause > 0:
                time.sleep(args.pause)

    print(f'\nSaved {len(rows)} question result(s) to {output.resolve()}')
    print('IMPORTANT: Dataset match counts are NOT audited precision; inspect unmatched names, aliases, and source completeness.')


if __name__ == '__main__':
    main()
