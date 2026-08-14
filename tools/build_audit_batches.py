"""Build vision-audit batch files from a project output dir.

    python tools/build_audit_batches.py <proj_out_dir> <status> <batch_out_dir> [chunk]

Each finding gets a globally-unique id "<drawingIdx>:<no>" so audit verdicts map back
to a specific drawing/crop (the flat-'no' collision made the first audit hard to
attribute). Writes b000.json.. and prints the count.
"""
import csv, os, json, sys

proj = sys.argv[1]
status = sys.argv[2]              # e.g. UNEXPECTED  or  NO_CHANGE_DETECTED,MISSED
out = sys.argv[3]
chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 6
wanted = set(status.split(","))

subs = sorted([d for d in os.listdir(proj) if os.path.isdir(os.path.join(proj, d))])
items = []
for di, sub in enumerate(subs):
    d = os.path.join(proj, sub)
    fp = os.path.join(d, "findings.csv")
    if not os.path.exists(fp):
        continue
    label = sub[3:] if sub[:2].isdigit() else sub
    for r in csv.DictReader(open(fp, encoding="utf-8")):
        if r["status"] not in wanted:
            continue
        no = r["no"]
        pidx = int(r["page"]) - 1
        tag = no if no.startswith("U") else "c" + no.split(".")[1]
        b = os.path.join(d, "crops", f"p{pidx}_{tag}_before.png")
        a = os.path.join(d, "crops", f"p{pidx}_{tag}_after.png")
        if not (os.path.exists(b) and os.path.exists(a)):
            continue
        items.append({"uid": f"{di}:{no}", "drawing": label, "no": no,
                      "status": r["status"], "text": r["text"],
                      "verified_value": r.get("verified_value", ""),
                      "page": r["page"], "before": os.path.abspath(b),
                      "after": os.path.abspath(a)})

os.makedirs(out, exist_ok=True)
for f in os.listdir(out):
    os.remove(os.path.join(out, f))
# batch by drawing, chunked
from collections import defaultdict
g = defaultdict(list)
for it in items:
    g[it["drawing"]].append(it)
n = 0
for drawing, lst in sorted(g.items()):
    for i in range(0, len(lst), chunk):
        p = os.path.join(out, f"b{n:03d}.json")
        json.dump({"drawing": drawing, "findings": lst[i:i + chunk]},
                  open(p, "w", encoding="utf-8"))
        n += 1
print(f"items={len(items)} batches={n}")
