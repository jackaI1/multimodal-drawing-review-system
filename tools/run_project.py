"""Run a whole-project (folder vs folder) comparison from the CLI.

    python tools/run_project.py <redlined_dir> <revised_dir> <out_dir> [--ocr] [--vlm]
"""
import sys, os, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from redlinecheck._compute import limit_threads
limit_threads(leave_free=1)
from redlinecheck.config import Config
from redlinecheck import project

red_dir, rev_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
red = sorted(glob.glob(os.path.join(red_dir, "*")))
rev = sorted(glob.glob(os.path.join(rev_dir, "*")))

cfg = Config()
verifier = None
if "--ocr" in sys.argv:
    cfg.use_ocr = True
    from redlinecheck import ocr
    verifier = ocr.make_verifier(cfg)
if "--vlm" in sys.argv:
    cfg.use_vlm = True
    from redlinecheck import vlm, ocr
    vs = [vlm.make_verifier(cfg)]
    if cfg.use_ocr:
        vs = [ocr.make_verifier(cfg)] + vs
    verifier = vlm.chained(*vs) if len(vs) > 1 else vs[0]

t = time.time()
def cb(d, n):
    print(f"  pair {d}/{n} done ({time.time()-t:.0f}s)", flush=True)

res = project.compare_project(red, rev, cfg, out, verifier=verifier, progress_cb=cb)
print("\n==== PROJECT RESULT ====")
print("pairs matched:", res["n_pairs"])
for p in res["pairs"]:
    print(f"   {p['redlined']}  ->  {p['revised']}   (score {p['score']})")
print("skipped:", res["skipped"])
print("counts:", res["counts"])
print("total finding rows:", len(res["rows"]))
print(f"TIME: {time.time()-t:.0f}s")
