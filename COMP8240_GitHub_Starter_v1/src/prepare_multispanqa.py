#!/usr/bin/env python3
"""Prepare a documented CLOSED-BOOK MultiSpanQA subset for CoVe.

Run on the public original MultiSpanQA JSON (prefer its labelled validation/test
split). Gold spans are extracted from BIO labels over CONTEXT tokens. The
context is used ONLY to construct reference answers, NEVER as a model input.

IMPORTANT: This deterministic subset is NOT known to be the exact 418 question
subset chosen by Dhuliawala et al. (ACL 2024). Report it as a reconstructed
MultiSpanQA factoid-style subset unless original subset IDs are recovered.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from datetime import datetime, timezone


def detok(parts):
    t = ' '.join(str(x) for x in parts).strip()
    t = re.sub(r'\s+([,.;:!?%\)\]\}])', r'\1', t)
    t = re.sub(r'([\(\[\{])\s+', r'\1', t)
    t = re.sub(r"\s+(['’]s|n't|'re|'ve|'m|'ll|'d)\b", r'\1', t)
    return re.sub(r'\s+', ' ', t).strip()


def bio_spans(tokens, labels):
    if not isinstance(tokens, list) or not isinstance(labels, list) or len(tokens) != len(labels):
        raise ValueError('context/label length mismatch')
    out, acc = [], []
    def commit():
        if acc:
            s = detok(acc)
            if s:
                out.append((s, len(acc)))
            acc.clear()
    for tok, label in zip(tokens, labels):
        l = str(label).upper().split('-')[0]
        if l == 'B':
            commit(); acc.append(tok)
        elif l == 'I':
            if not acc:
                raise ValueError('invalid BIO: I without B')
            acc.append(tok)
        elif l == 'O':
            commit()
        else:
            raise ValueError(f'unrecognized BIO label {label!r}')
    commit()
    return out


def load_records(path):
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() == '.jsonl':
        data = [json.loads(s) for s in text.splitlines() if s.strip()]
    else:
        data = json.loads(text)
    if isinstance(data, dict):
        if isinstance(data.get('data'), list):
            data = data['data']
        elif isinstance(data.get('examples'), list):
            data = data['examples']
        else:
            raise ValueError('Expected JSON list of MultiSpanQA records or {data:[...]}')
    if not isinstance(data, list):
        raise ValueError('Source file must contain a JSON list')
    return data


def convert(raw, min_spans=2, max_tokens=3):
    q = raw.get('question')
    if isinstance(q, list):
        question = detok(q)
    elif isinstance(q, str):
        question = q.strip()
    else:
        return None, 'bad_question'
    ctx = raw.get('context')
    tags = raw.get('label')
    if not isinstance(ctx, list) or not isinstance(tags, list):
        return None, 'missing_context_or_labels'
    try:
        spans = bio_spans(ctx, tags)
    except ValueError:
        return None, 'invalid_bio'
    # Gold may repeat the same text at different locations; keep one distinct answer.
    unique, seen = [], set()
    for value, length in spans:
        k = value.casefold()
        if k not in seen:
            seen.add(k); unique.append((value, length))
    if len(unique) < min_spans:
        return None, 'too_few_spans'
    if any(length > max_tokens for _, length in unique):
        return None, 'long_gold_span'
    if not question:
        return None, 'empty_question'
    return {
        'id': str(raw.get('id') or raw.get('_id') or hashlib.sha256(question.encode()).hexdigest()[:20]),
        'question': question,
        'gold_spans': [value for value, _ in unique],
        'gold_span_token_lengths': [length for _, length in unique],
        'num_gold_spans': len(unique),
        'source_type': raw.get('type'),
        'source_structure': raw.get('structure'),
        'closed_book': True,
    }, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Original MultiSpanQA JSON (with context+BIO gold labels)')
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=60, help='Number of candidates to freeze, 0 = all')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-span-tokens', type=int, default=3)
    p.add_argument('--min-spans', type=int, default=2)
    args = p.parse_args()
    src, out = Path(args.input), Path(args.output)
    if not src.is_file():
        p.error(f'File does not exist: {src}')
    if args.limit < 0 or args.max_span_tokens < 1 or args.min_spans < 1:
        p.error('Invalid limit or span-length bounds')
    data = load_records(src)
    collected, excluded = [], {}
    seen_ids = set()
    for raw in data:
        if not isinstance(raw, dict):
            excluded['bad_record'] = excluded.get('bad_record', 0)+1
            continue
        row, reason = convert(raw, args.min_spans, args.max_span_tokens)
        if reason:
            excluded[reason] = excluded.get(reason, 0)+1
        elif row['id'] in seen_ids:
            excluded['duplicate_id'] = excluded.get('duplicate_id', 0)+1
        else:
            seen_ids.add(row['id']); collected.append(row)
    if not collected:
        raise SystemExit('No eligible examples. Confirm this is labelled MultiSpanQA data; test data may not include gold labels.')
    rng = random.Random(args.seed)
    rng.shuffle(collected)
    kept = collected[:args.limit] if args.limit else collected
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_file': str(src.resolve()), 'source_sha256': sha,
        'source_rows': len(data), 'eligible_rows': len(collected),
        'written_rows': len(kept), 'excluded_counts': excluded,
        'selection': {'seed': args.seed, 'min_spans':args.min_spans,
                      'max_span_tokens':args.max_span_tokens, 'limit':args.limit},
        'exact_original_418_subset': False,
        'note': 'Filter approximates the published <=3 tokens/span criterion; original factoid and 418 ID selection are not publicly confirmed. Context withheld from model.'
    }
    report = out.with_suffix('.manifest.json')
    report.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Input: {len(data)} rows; eligible: {len(collected)}; saved: {len(kept)}')
    print(f'Output: {out.resolve()}')
    print(f'Manifest: {report.resolve()}')
    print('First 3 candidate examples (review for factoid/closed-book suitability):')
    for r in kept[:3]:
        print(' -', r['id'], r['question'], '=>', r['gold_spans'])
    print('IMPORTANT: This is a reconstructed subset, NOT verified as the paper\'s exact 418 questions.')

if __name__ == '__main__':
    main()
