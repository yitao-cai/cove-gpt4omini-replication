#!/usr/bin/env python3
"""Run a small paired baseline/CoVe MultiSpanQA pilot via the user's community code.

Copy into chain-of-verification/src alongside cove_chains.py, prompts.py and the
closed-book execute_verification_chain.py (from the user's tested setup).

Baseline is the EXACT baseline_response generated inside the CoVe chain,
preventing unnecessary rerun variation. We keep raw strings for later auditing.
This script DOES NOT claim F1: output parsing and gold comparison are separate.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / '.env')
load_dotenv(ROOT / '.env')


def load(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    for row in rows:
        if not isinstance(row.get('gold_spans'), list) or not isinstance(row.get('question'), str):
            raise ValueError('Input needs question and gold_spans from prepare_multispanqa.py')
    return rows


def run_one(chain, row, params):
    try:
        r = chain.invoke({'original_question': row['question']})
        required = ['baseline_response', 'verification_questions', 'verification_answers', 'final_answer']
        missing = [k for k in required if not isinstance(r.get(k), str)]
        if missing:
            raise ValueError(f'Missing chain outputs: {missing}')
        return {
            'id': row['id'], 'question':row['question'], 'gold_spans':row['gold_spans'],
            'baseline_response':r['baseline_response'],
            'verification_questions':r['verification_questions'],
            'verification_answers':r['verification_answers'],
            'final_answer':r['final_answer'],
            'status':'ok','settings': params,
            'ended_at_utc':datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            'id':row['id'],'question':row['question'],'gold_spans':row['gold_spans'],
            'status':'error','error':repr(e),'settings':params,
            'ended_at_utc':datetime.now(timezone.utc).isoformat()
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=5, help='Start with 5; 0 means all')
    p.add_argument('--model', default='gpt-4o-mini')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--max-tokens', type=int, default=800)
    p.add_argument('--pause', type=float, default=1.0)
    p.add_argument('--dry-run', action='store_true', help='Check input; do not call API')
    args = p.parse_args()
    if args.limit < 0:
        p.error('--limit must be nonnegative')
    dataset = Path(args.dataset)
    if not dataset.is_file():
        p.error(f'Dataset not found: {dataset}')
    rows = load(dataset)
    chosen = rows if args.limit == 0 else rows[:args.limit]
    if not chosen:
        p.error('No selected questions')
    out = Path(args.output)
    existing = set()
    if out.exists():
        for line in out.read_text(encoding='utf-8').splitlines():
            if line.strip():
                prior = json.loads(line)
                if prior.get('status') == 'ok':
                    existing.add(str(prior['id']))
    todo = [r for r in chosen if str(r['id']) not in existing]
    print(f'Loaded {len(rows)} candidates; selected {len(chosen)}, previously OK {len(existing)}, to run {len(todo)}')
    for row in chosen[:3]:
        print(' ', row['id'], row['question'], ' | gold:', row['gold_spans'])
    if args.dry_run:
        print('DRY RUN OK. No API calls made.')
        return
    if not os.getenv('OPENAI_API_KEY'):
        p.error('OPENAI_API_KEY missing; set locally (never upload the key)')
    from langchain_openai import ChatOpenAI
    from cove_chains import MultiSpanCOVEChain
    from execute_verification_chain import ExecuteVerificationChain
    llm = ChatOpenAI(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    chain = MultiSpanCOVEChain(llm)()
    executors = [sub for sub in chain.chains if isinstance(sub, ExecuteVerificationChain)]
    if len(executors) != 1 or executors[0].use_search_tool:
        raise RuntimeError('Expected exactly one closed-book verification executor; stop to inspect chain.')
    params = {'model_requested':args.model, 'temperature':args.temperature,
              'max_tokens':args.max_tokens, 'verification':'community_factored_closed_book',
              'dataset_file':str(dataset.resolve())}
    out.parent.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(todo, 1):
        print(f'[{i}/{len(todo)}] {row["id"]} — {row["question"]}', flush=True)
        result = run_one(chain, row, params)
        with out.open('a',encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False)+'\n')
        if result['status']=='ok':
            print('  BASELINE:', result['baseline_response'][:180].replace('\n', ' '))
            print('  COVE:    ', result['final_answer'][:180].replace('\n', ' '))
        else:
            print('  ERROR:', result['error'], file=sys.stderr)
            print('Stopping to avoid costly repeated calls; review error and rerun to resume.')
            break
        if i < len(todo) and args.pause>0:
            time.sleep(args.pause)
    print('Saved/resumable:', out.resolve())
    print('No F1 calculated yet; raw answers must first be normalized and audited against gold spans.')

if __name__=='__main__':
    main()
