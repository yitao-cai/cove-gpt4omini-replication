#!/usr/bin/env python3
"""Complete Wikidata Gold lists with resumable SPARQL pages (no wbgetentities API).

Designed for build_dataset.py v2 used in the COMP8240 CoVe project.

Exact Gold rules:
  P31=human, P106=occupation, P19=birthplace,
  birthplace P131? city (zero or one hop), sitelinks>=min_sitelinks.

Important:
  * This is a reconstructed benchmark, not the original paper's 56-question set.
  * It retains the source dataset's near-miss entities, removing any QIDs that
    overlap the rebuilt Gold list; near-miss completeness is NOT audited.
  * A valid contact URL/email is required in the User-Agent (Wikimedia policy).
  * Each successful page is cached immediately. If rate-limited, stop and rerun
    the SAME command after the Retry-After interval; cached pages are reused.
  * Source JSONL is never overwritten.

Run from the directory holding near_miss_questions.jsonl:
  python rebuild_wikidata_safe.py --dataset near_miss_questions.jsonl \
    --output chicago_gold_complete.jsonl --report chicago_coverage_rebuilt.json \
    --contact https://github.com/YOUR-USERNAME/YOUR-REPOSITORY
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

WDQS = "https://query.wikidata.org/sparql"
QID_RE = re.compile(r"Q[1-9][0-9]*\Z")
RETRYABLE = {500, 502, 503, 504}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def atomic_jsonl(path, objs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def valid_qid(q):
    if not isinstance(q, str) or not QID_RE.fullmatch(q):
        raise ValueError(f"Invalid QID: {q!r}")
    return q


def pattern(item):
    city = valid_qid(item["city_qid"])
    occupation = valid_qid(item["occupation_qid"])
    min_sl = int(item.get("min_sitelinks", 5))
    if min_sl < 0:
        raise ValueError("min_sitelinks must be >= 0")
    return (f"?person wdt:P31 wd:Q5 ;\n"
            f"        wdt:P106 wd:{occupation} ;\n"
            f"        wdt:P19 ?bp ;\n"
            f"        wikibase:sitelinks ?sl .\n"
            f"?bp wdt:P131? wd:{city} .\n"
            f"FILTER(?sl >= {min_sl})")


def remote_count_query(item):
    return "SELECT (COUNT(DISTINCT ?person) AS ?n) WHERE {\n" + pattern(item) + "\n}"


def page_query(item, page_size, offset):
    # Fetch labels in the SAME SPARQL results; avoid the separate wbgetentities
    # Action API, which was where the user's previous run received HTTP 429.
    return ("SELECT DISTINCT ?person ?bp ?personLabel ?bpLabel WHERE {\n"
            + pattern(item) + "\n"
            + "SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\". }\n"
            + "}\nORDER BY ?person ?bp\n"
            + f"LIMIT {page_size}\nOFFSET {offset}")


class RateLimited(RuntimeError):
    pass


def query_rows(session, query, user_agent, retries=2):
    for attempt in range(retries + 1):
        try:
            r = session.get(WDQS, params={"query": query, "format": "json"},
                            headers={"User-Agent": user_agent,
                                     "Accept": "application/sparql-results+json",
                                     "Accept-Encoding": "gzip,deflate"},
                            timeout=90)
        except requests.RequestException as exc:
            if attempt >= retries:
                raise RuntimeError(f"Network error after retries: {exc}") from exc
            wait = 5 * (attempt + 1)
            print(f"    Network error ({type(exc).__name__}); waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r.json()["results"]["bindings"]
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After", "not provided")
            raise RateLimited("Wikidata returned HTTP 429. Stop requests and retry later. "
                              f"Retry-After: {retry_after}. Saved pages can be reused.")
        if r.status_code not in RETRYABLE:
            r.raise_for_status()
        if attempt >= retries:
            raise RuntimeError(f"Wikidata returned HTTP {r.status_code} after retries")
        wait = 5 * (attempt + 1)
        print(f"    HTTP {r.status_code}; waiting {wait}s", flush=True)
        time.sleep(wait)
    raise AssertionError("unreachable")


def cached_query(session, query, path, user_agent):
    fingerprint = hashlib.sha256(query.encode("utf-8")).hexdigest()
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("query_sha256") == fingerprint and isinstance(data.get("rows"), list):
            return data["rows"], True
    rows = query_rows(session, query, user_agent)
    atomic_json(path, {"fetched_utc": utc_now(), "query_sha256": fingerprint,
                       "rows": rows})
    return rows, False


def as_qid(binding):
    value = binding["value"]
    return valid_qid(value.rsplit("/", 1)[-1])


def label(row, field, qid):
    result = row.get(field, {}).get("value")
    return result if result and result != qid else qid


def rebuild_item(session, item, cache, page_size, delay, user_agent):
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", item["id"])
    count_query = remote_count_query(item)
    count_rows, from_cache = cached_query(session, count_query,
                                           cache / f"{slug}_count.json", user_agent)
    if not from_cache:
        time.sleep(delay)
    expected = int(count_rows[0]["n"]["value"])
    print(f"  WDQS expected unique Gold QIDs: {expected}", flush=True)
    persons = {}
    offset = 0
    pages = 0
    while True:
        query = page_query(item, page_size, offset)
        page_path = cache / f"{slug}_page_{offset:06d}.json"
        rows, from_cache = cached_query(session, query, page_path, user_agent)
        if not from_cache:
            time.sleep(delay)
        for row in rows:
            person = as_qid(row["person"])
            birth_place = as_qid(row["bp"])
            entry = persons.setdefault(person,
                        {"qid": person, "name": label(row, "personLabel", person),
                         "birthplace_qids": [], "_bp_labels": {}})
            if entry["name"] == person:
                entry["name"] = label(row, "personLabel", person)
            entry["_bp_labels"][birth_place] = label(row, "bpLabel", birth_place)
            if birth_place not in entry["birthplace_qids"]:
                entry["birthplace_qids"].append(birth_place)
        pages += 1
        print(f"    offset={offset}, rows={len(rows)}, unique={len(persons)}"
              + (" [cached]" if from_cache else ""), flush=True)
        if len(rows) < page_size:
            break
        offset += page_size
    results = []
    for q in sorted(persons, key=lambda s: int(s[1:])):
        p = persons[q]
        bps = sorted(p["birthplace_qids"], key=lambda s: int(s[1:]))
        results.append({"qid": q, "name": p["name"],
                        "born_in": "; ".join(p["_bp_labels"].get(bp, bp) for bp in bps),
                        "birthplace_qids": bps})
    named = sum(p["name"] != p["qid"] for p in results)
    print(f"  Complete pages: {pages}; unique={len(results)}, named={named}/{len(results)}", flush=True)
    return expected, results, pages


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="near_miss_questions.jsonl")
    parser.add_argument("--output", default="chicago_gold_complete.jsonl")
    parser.add_argument("--report", default="chicago_coverage_rebuilt.json")
    parser.add_argument("--cache-dir", default=".wikidata_gold_page_cache")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--contact", default=os.environ.get("WIKIDATA_CONTACT"),
                        help="Your real public project URL or contact email for Wikimedia User-Agent policy")
    args = parser.parse_args()
    if args.page_size < 1 or args.page_size > 500 or args.delay < 0:
        parser.error("page-size must be 1..500 and delay nonnegative")
    if not args.contact or not args.contact.strip():
        parser.error("Supply --contact with a REAL contact URL/email (Wikimedia User-Agent policy)")
    inp, out, rep = (Path(p).resolve() for p in (args.dataset, args.output, args.report))
    if inp == out:
        parser.error("--output must not overwrite --dataset")
    with inp.open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if not items:
        parser.error("Dataset is empty")
    cache = Path(args.cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    ua = f"COMP8240-CoVe-Wikidata-Rebuild/2.0 ({args.contact.strip()}) python-requests"
    report = {"generated_utc": utc_now(),
              "rules": "P31 Q5, P106 occupation, P19 birthplace, P131? city, sitelinks >= threshold; no English Wikipedia filter",
              "source": str(inp), "note": "Reconstructed dataset, not original paper's 56-question benchmark",
              "results": []}
    rebuilt_items = []
    with requests.Session() as session:
        for item in items:
            print(f"\n{item['id']}:", flush=True)
            try:
                expected, gold, pages = rebuild_item(session, item, cache, args.page_size,
                                                      args.delay, ua)
            except (RateLimited, RuntimeError) as exc:
                print(f"\nSTOPPED: {exc}", file=sys.stderr)
                print("Pages saved in cache. Do NOT delete cache; rerun the same command after rate limit clears.", file=sys.stderr)
                sys.exit(2)
            obtained = len(gold)
            gold_ids = {p["qid"] for p in gold}
            distractors = [p for p in item.get("distractors", []) if p.get("qid") not in gold_ids]
            removed_overlap = len(item.get("distractors", [])) - len(distractors)
            updated = dict(item)
            updated["gold"] = gold
            updated["distractors"] = distractors
            updated["gold_rebuild"] = {
                "retrieved_utc": utc_now(), "distinct_remote_count": expected,
                "obtained_unique_gold": obtained,
                "count_matches": obtained == expected,
                "distractor_overlap_removed": removed_overlap,
                "distractor_completeness_checked": False,
                "notes": "Current source dataset's original retrieved date refers to distractors; Gold rebuilt separately",
            }
            report["results"].append({
                "id": item["id"], "local_unique_gold": len({p["qid"] for p in item["gold"]}),
                "remote_distinct_count": expected, "rebuilt_unique_gold": obtained,
                "match": obtained == expected,
                "unnamed_qids": sum(p["name"] == p["qid"] for p in gold),
                "overlapping_distractors_removed": removed_overlap,
                "pages": pages,
            })
            rebuilt_items.append(updated)
    # Only publish completed results after all questions have been fetched.
    # Counts may differ because Wikidata changed between requests; flag these.
    atomic_jsonl(out, rebuilt_items)
    atomic_json(rep, report)
    print(f"\nSaved rebuilt dataset: {out}")
    print(f"Saved audit report:    {rep}")
    if any(not x["match"] or x["unnamed_qids"] for x in report["results"]):
        print("WARNING: Some counts/labels are incomplete. Review the audit report before scoring.")
    else:
        print("All QID counts match; still review labels and sampled birthplaces manually.")


if __name__ == "__main__":
    main()
