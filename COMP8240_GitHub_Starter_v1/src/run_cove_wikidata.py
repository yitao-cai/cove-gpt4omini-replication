#!/usr/bin/env python3
"""Small *factored-style* CoVe smoke test on build_dataset.py JSONL.

This is a teaching / sanity-check runner, not the authors' exact prompt reproduction.
Run with: python run_cove_wikidata.py --dataset near_miss_questions.jsonl --id boston-actors
Requirements: python -m pip install openai, and OPENAI_API_KEY environment variable.
Output is one JSON object per example; inspect unmatched names manually.
"""
import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path


def normalize(text):
    """Provisional matching only; aliases/homonyms still require manual audit."""
    return re.sub(r"[^\w]", "", str(text).casefold())


def names_from(reply, field="names"):
    value = reply.get(field)
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"Expected JSON field {field!r} to be a list of names: {reply!r}")
    seen = set()
    result = []
    for name in value:
        key = normalize(name)
        if key and key not in seen:
            seen.add(key)
            result.append(name.strip())
    return result


def score(names, example):
    """Unmatched names are UNKNOWN, not necessarily hallucinations.

    Each label is a provisional Wikidata-based classification, not human-verified truth.
    """
    gold = {normalize(p["name"]): p for p in example["gold"]}
    near = {normalize(p["name"]): p for p in example["distractors"]}
    classified = {"gold": [], "near_miss": [], "unmatched": []}
    for name in names:
        k = normalize(name)
        if k in gold:
            classified["gold"].append({"name": name, "born_in": gold[k]["born_in"]})
        elif k in near:
            classified["near_miss"].append({"name": name, "born_in": near[k]["born_in"]})
        else:
            classified["unmatched"].append({"name": name, "manual_review_required": True})
    n = len(names)
    g, u = len(classified["gold"]), len(classified["unmatched"])
    classified["total"] = n
    classified["provisional_correct_lower_bound"] = round(g / n, 4) if n else None
    classified["provisional_correct_upper_bound"] = round((g + u) / n, 4) if n else None
    return classified


def call_json(client, model, system, user):
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system + " Return valid JSON only."},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("API returned an empty response")
    return json.loads(content), {
        "request_id": response.id,
        "resolved_model": response.model,
        "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
        "completion_tokens": response.usage.completion_tokens if response.usage else None,
    }


def run_one(client, model, example, max_names):
    question = example["question"]
    calls = []
    print(f"\nQUESTION: {question}", flush=True)

    # Step 1: baseline; also serves as CoVe's initial draft (one-to-one comparison).
    baseline, meta = call_json(
        client, model,
        "Answer factual entity-list questions from your internal knowledge only. No browsing. ",
        f"{question}\nReturn JSON {{\"names\": [\"Full Name\", ...]}} "
        f"with at most {max_names} distinct people. If unsure, list fewer. "
        "Use only names, no birthplaces or explanations.",
    )
    calls.append({"stage": "baseline", **meta})
    initial_names = names_from(baseline)
    if len(initial_names) > max_names:
        raise ValueError(f"Baseline exceeded max_names={max_names}: {initial_names}")
    print("BASELINE:", initial_names, flush=True)

    # Step 2: independently plan questions for *every* baseline claim.
    if initial_names:
        plan, meta = call_json(
            client, model,
            "Plan simple, atomic, factual verification questions. Do not answer them.",
            "Original user question: " + question + "\n"
            "Names in initial draft: " + json.dumps(initial_names, ensure_ascii=False) + "\n"
            "For EACH name, ask exactly one stand-alone neutral question about the person's "
            "city of birth. Do not mention the target city in your verification question; "
            "do not include the original answer in a question. "
            "Return JSON {\"questions\": [{\"name\": \"exact initial name\", "
            "\"question\": \"Which city was ... born in?\"}, ...]}.",
        )
        calls.append({"stage": "planning", **meta})
        entries = plan.get("questions")
        if not isinstance(entries, list):
            raise ValueError("Planning result lacks 'questions' array")
        by_name = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not isinstance(entry.get("question"), str):
                raise ValueError(f"Invalid verification question: {entry}")
            k = normalize(entry["name"])
            if k in by_name:
                raise ValueError(f"Duplicate verification name: {entry['name']}")
            by_name[k] = entry
        if set(by_name) != {normalize(x) for x in initial_names}:
            raise ValueError("Planning missed/added a baseline name; inspect prompt and rerun")
    else:
        plan = {"questions": []}
        by_name = {}

    # Step 3: isolated calls: NO baseline, target-city, original question or past QA in context.
    verifications = []
    for name in initial_names:
        item = by_name[normalize(name)]
        answer, meta = call_json(
            client, model,
            "Answer the following standalone factual question from your internal knowledge only. "
            "If unknown, explicitly say unknown; do not guess or browse.",
            item["question"] + "\nReturn JSON {\"birthplace\": \"city or unknown\", "
            "\"confidence\": \"known or uncertain\"}.",
        )
        calls.append({"stage": "verification", "name": name, **meta})
        verifications.append({"name": name, "question": item["question"], "answer": answer})
        print(f"VERIFY {name}: {answer}", flush=True)

    # Step 4: revise using checks; no external source used in answer generation.
    if initial_names:
        final, meta = call_json(
            client, model,
            "Revise a list of entities using the independently answered verification questions. "
            "Avoid unsupported names; do not invent new names.",
            "Original question: " + question + "\n"
            "Initial names: " + json.dumps(initial_names, ensure_ascii=False) + "\n"
            "Independent checks: " + json.dumps(verifications, ensure_ascii=False) + "\n"
            "Keep only the initial names supported by the independent checks as meeting "
            "the location requirement. You may omit uncertain entries. "
            "Return JSON {\"names\": [\"Full Name\", ...]}.",
        )
        calls.append({"stage": "revision", **meta})
    else:
        final = {"names": []}
    final_names = names_from(final)
    extras = [x for x in final_names if normalize(x) not in {normalize(n) for n in initial_names}]
    if extras:
        raise ValueError(f"Revision unexpectedly added names: {extras}; check prompt compliance")
    print("FINAL:   ", final_names, flush=True)

    baseline_score = score(initial_names, example)
    final_score = score(final_names, example)
    print("LABEL COUNTS (provisional, manual audit required):")
    for label, result in (("baseline", baseline_score), ("CoVe", final_score)):
        print(f"  {label:<8} gold={len(result['gold'])}, near={len(result['near_miss'])}, "
              f"unmatched={len(result['unmatched'])}, total={result['total']}")

    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_requested": model,
        "temperature": 0,
        "method": "teaching_smoke_test_factored_style_cove_NOT_exact_paper_prompts",
        "dataset_id": example["id"], "dataset_retrieved": example.get("retrieved"),
        "question": question,
        "baseline": {"names": initial_names, "score": baseline_score},
        "verification_plan": plan,
        "independent_verifications": verifications,
        "cove": {"names": final_names, "score": final_score},
        "api_calls": calls,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="near_miss_questions.jsonl")
    p.add_argument("--id", default="boston-actors", help="Dataset question ID")
    p.add_argument("--limit", type=int, default=1, help="How many questions to process, default 1")
    p.add_argument("--max-names", type=int, default=10)
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    p.add_argument("--output", default="cove_smoke_results.jsonl")
    args = p.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Use $env:OPENAI_API_KEY='...' in PowerShell.")
    from openai import OpenAI
    path = Path(args.dataset)
    if not path.is_file():
        sys.exit(f"Dataset not found: {path.resolve()} (check your PowerShell working directory)")
    examples = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if args.id == "all":
        examples = examples[:args.limit]
    else:
        examples = [x for x in examples if x["id"] == args.id][:args.limit]
    if not examples:
        sys.exit(f"No examples with id={args.id!r}. Available ids: " + ", ".join(
            x["id"] for x in [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        ))
    client = OpenAI(timeout=60.0, max_retries=2)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Using {args.model}; reading {path.resolve()}; writing {out.resolve()}")
    with out.open("w", encoding="utf-8") as handle:
        for example in examples:
            record = run_one(client, args.model, example, args.max_names)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    print(f"\nSaved {len(examples)} result(s): {out.resolve()}")
    print("IMPORTANT: Unmatched entries must be manually checked before final precision is reported.")


if __name__ == "__main__":
    main()
