import csv
import glob
import json
import statistics
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from quality import keyword_score, make_judge

load_dotenv()
JUDGE_DELAY_S = 4.0     # free-tier rate limit

questions = {}
with open("data/questions.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            q = json.loads(line)
            questions[q["id"]] = q

csv_path = sorted(glob.glob("results/run_*.csv"))[-1]          # newest run
with open(csv_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"Scoring {csv_path}\n")

# Check: with temperature=0, do repeats give the same answer?
for model in dict.fromkeys(r["model"] for r in rows):
    changed = [i for i in questions
               if len({r["answer"] for r in rows if r["model"] == model and r["item_id"] == i}) > 1]
    print(f"{model}: answers differed across repeats on {len(changed)} question(s) {changed}")
print()

judge = make_judge(ChatGroq(model="openai/gpt-oss-120b", temperature=0, reasoning_effort="medium"))

for r in rows:
    q = questions[r["item_id"]]
    r["keyword_score"] = keyword_score(r["answer"], q["keywords"])
    r["judge_score"], r["judge_reason"] = "", ""
    if r["repeat"] == "0":                          # judge one copy of each answer
        time.sleep(JUDGE_DELAY_S)
        r["judge_score"], r["judge_reason"] = judge(q, r["answer"])
        print(f"{r['model']:<20} {r['item_id']:<14} kw={r['keyword_score']:4.0%}  "
              f"judge={r['judge_score']}  {r['judge_reason']}")

out_path = csv_path.replace("run_", "scored_")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {out_path}\n")
print(f"{'model':<20} {'keyword acc':>12} {'judge (1-5)':>12}")
for model in dict.fromkeys(r["model"] for r in rows):
    r_m = [r for r in rows if r["model"] == model]
    kw = statistics.mean(r["keyword_score"] for r in r_m)
    js = [r["judge_score"] for r in r_m if isinstance(r["judge_score"], int)]
    print(f"{model:<20} {kw:12.0%} {statistics.mean(js) if js else 0:12.2f}")
