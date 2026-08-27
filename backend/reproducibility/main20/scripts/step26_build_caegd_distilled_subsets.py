from __future__ import annotations
import hashlib, json, math, random, re
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260813
RATIOS = (0.25, 0.50, 0.75)

HERE = Path(__file__).resolve().parent
TRAIN_FILE = HERE / "04_main20k_split" / "train.jsonl"
OUT_DIR = HERE / "07_caegd_distillation"

CAL_RE = re.compile(r"- Average calories:\s*(\d+(?:\.\d+)?)\s*kcal", re.I)
PRO_RE = re.compile(r"- Average protein:\s*(\d+(?:\.\d+)?)\s*g", re.I)
FIB_RE = re.compile(r"- Average fibre:\s*(\d+(?:\.\d+)?)\s*g", re.I)
MATCH_WEIGHT = {"full": 1.0, "weak": 2.0, "partial": 3.0}


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def md(row):
    x = row.get("metadata", {})
    return x if isinstance(x, dict) else {}


def actuals(text):
    def get(rx):
        m = rx.search(text)
        return float(m.group(1)) if m else None
    return {"calories": get(CAL_RE), "protein": get(PRO_RE), "fiber": get(FIB_RE)}


def boundary(row):
    c = md(row).get("constraints", {}) or {}
    a = actuals(str(row.get("input", "")))
    pairs = [(a["calories"], c.get("max_calories")), (a["protein"], c.get("min_protein"))]
    if c.get("min_fiber") is not None:
        pairs.append((a["fiber"], c.get("min_fiber")))
    vals = []
    for av, tv in pairs:
        if av is None or tv is None:
            continue
        tv = float(tv)
        rel = abs(float(av) - tv) / max(abs(tv), 1.0)
        vals.append(1.0 / (1.0 + 5.0 * rel))
    return sum(vals) / len(vals) if vals else 0.0


def failed_count(row):
    checks = md(row).get("constraint_checks", {}) or {}
    return sum(v is False for v in checks.values())


def tie(sample_id):
    h = hashlib.sha256(f"{SEED}:{sample_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def utility(row, restaurant_freq, query_freq):
    m = md(row)
    match = str(m.get("match_type", "full"))
    restaurant = str(m.get("restaurant_name", ""))
    query_id = str(m.get("query_id", ""))
    rare_r = 1.0 / math.sqrt(max(restaurant_freq[restaurant], 1))
    rare_q = 1.0 / math.sqrt(max(query_freq[query_id], 1))
    return (
        1.20 * MATCH_WEIGHT.get(match, 1.0)
        + 1.50 * boundary(row)
        + 0.45 * failed_count(row)
        + 0.80 * rare_r
        + 0.30 * rare_q
        + 0.01 * tie(str(row.get("sample_id", "")))
    )


def order_bucket(rows, restaurant_freq, query_freq):
    remaining = list(rows)
    ordered = []
    seen_r, seen_q, seen_m = set(), set(), set()
    while remaining:
        best_i, best_s = 0, float("-inf")
        for i, row in enumerate(remaining):
            m = md(row)
            r = str(m.get("restaurant_name", ""))
            q = str(m.get("query_id", ""))
            mt = str(m.get("match_type", ""))
            s = utility(row, restaurant_freq, query_freq)
            if r not in seen_r:
                s += 1.00
            if q not in seen_q:
                s += 0.55
            if mt not in seen_m:
                s += 0.75
            if s > best_s:
                best_i, best_s = i, s
        chosen = remaining.pop(best_i)
        m = md(chosen)
        seen_r.add(str(m.get("restaurant_name", "")))
        seen_q.add(str(m.get("query_id", "")))
        seen_m.add(str(m.get("match_type", "")))
        ordered.append(chosen)
    return ordered


def coverage(rows):
    return {
        "signatures": len({str(md(r).get("constraint_signature_id", "")) for r in rows} - {""}),
        "queries": len({str(md(r).get("query_id", "")) for r in rows} - {""}),
        "restaurants": len({str(md(r).get("restaurant_name", "")) for r in rows} - {""}),
        "match": dict(Counter(str(md(r).get("match_type", "unknown")) for r in rows)),
    }


def main():
    rows = read_jsonl(TRAIN_FILE)
    if len(rows) != 15983:
        raise RuntimeError(f"Expected 15983 train samples, got {len(rows)}")

    by_sig = defaultdict(list)
    restaurant_freq, query_freq = Counter(), Counter()
    for row in rows:
        m = md(row)
        sid = str(m.get("constraint_signature_id", ""))
        if not sid:
            raise RuntimeError("Missing constraint_signature_id")
        by_sig[sid].append(row)
        restaurant_freq[str(m.get("restaurant_name", ""))] += 1
        query_freq[str(m.get("query_id", ""))] += 1

    if len(by_sig) != 500:
        raise RuntimeError(f"Expected 500 signatures, got {len(by_sig)}")

    ordered = {
        sid: order_bucket(bucket, restaurant_freq, query_freq)
        for sid, bucket in by_sig.items()
    }

    rng = random.Random(SEED)
    sigs = sorted(ordered)
    ranking = []
    for round_idx in range(max(len(v) for v in ordered.values())):
        active = [s for s in sigs if round_idx < len(ordered[s])]
        rng.shuffle(active)
        ranking.extend(ordered[s][round_idx] for s in active)

    if len(ranking) != len(rows):
        raise RuntimeError("Ranking size mismatch")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "Constraint-Aware Evidence-Grounded Data Distillation (CAEGD)",
        "seed": SEED,
        "source_train_samples": len(rows),
        "source_train_signatures": len(by_sig),
        "subsets": {},
    }
    subset_ids = {}

    for ratio in RATIOS:
        n = round(len(rows) * ratio)
        subset = ranking[:n]
        label = f"{int(ratio*100)}pct"
        path = OUT_DIR / f"train_caegd_{label}.jsonl"
        write_jsonl(path, subset)
        subset_ids[label] = {str(r.get("sample_id", "")) for r in subset}
        summary["subsets"][label] = {"samples": len(subset), **coverage(subset), "file": path.name}

    full_ids = {str(r.get("sample_id", "")) for r in rows}
    summary["nested_checks"] = {
        "25_in_50": subset_ids["25pct"] <= subset_ids["50pct"],
        "50_in_75": subset_ids["50pct"] <= subset_ids["75pct"],
        "75_in_100": subset_ids["75pct"] <= full_ids,
    }

    summary_file = OUT_DIR / "caegd_distillation_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print("CAEGD DISTILLATION COMPLETE")
    print("=" * 72)
    print("100% train :", len(rows))
    for label in ("75pct", "50pct", "25pct"):
        s = summary["subsets"][label]
        print(f"{label:<5}: {s['samples']} samples | {s['signatures']} signatures | "
              f"{s['queries']} queries | {s['restaurants']} restaurants | {s['match']}")
    print("Nested checks:", summary["nested_checks"])
    print("Summary:", summary_file)


if __name__ == "__main__":
    main()
