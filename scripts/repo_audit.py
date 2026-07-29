"""
Full repo audit: security, stale files, code consistency, output consistency,
reproducibility, data hygiene. Finds anything that would embarrass us in review.
"""

import os, json, subprocess, sys, time
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
results = []

def check(name, condition, evidence):
    status = "PASS" if condition else "FAIL"
    results.append((status, name, evidence))
    print(f"  [{status}] {name}: {evidence}")

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, cwd=str(repo), text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        return e.output

print("=" * 80)
print("FULL REPO AUDIT")
print("=" * 80)

# ============================================================
# 1. SECURITY: .env not in git, no API keys in tracked files
# ============================================================
print("\n--- 1. Security ---")

gitignore = (repo / ".gitignore").read_text() if (repo / ".gitignore").exists() else ""
check(".env in .gitignore", ".env" in gitignore, f".gitignore contains .env: {'.env' in gitignore}")

log_env = run("git log --all --oneline -- .env")
check(".env never committed", "fatal" in log_env or len(log_env.strip()) == 0,
      f"git log for .env: {'clean' if not log_env.strip() else log_env.strip()[:100]}")

grep_key = run('git grep -i "nvapi-" -- "*.py" "*.md" "*.txt" "*.json" "*.env*" 2>nul || echo NONE')
check("No nvapi- keys in tracked files", "NONE" in grep_key or len(grep_key.strip()) == 0,
      f"nvapi- search: {grep_key.strip()[:100] if grep_key.strip() else 'none found'}")

grep_key2 = run('git grep -iE "nvapi-[A-Za-z0-9]" -- "*.py" "*.md" "*.txt" "*.json" "*.cfg" "*.ini" 2>nul || echo NONE')
check("No hardcoded real API keys in tracked files", "NONE" in grep_key2 or len(grep_key2.strip()) == 0,
      f"hardcoded key search: {grep_key2.strip()[:200] if grep_key2.strip() else 'none'}")

# Check .env file itself is gitignored
env_tracked = run("git ls-files --error-unmatch .env 2>nul && echo TRACKED || echo UNTRACKED")
check(".env is untracked (gitignored)", "UNTRACKED" in env_tracked, f"status: {env_tracked.strip()}")

# ============================================================
# 2. STALE FILES: files that shouldn't be in the repo
# ============================================================
print("\n--- 2. Stale / stray files ---")

stale_candidates = [
    "walkthrough.md",
    "implementation_plan.md",
    "task.md",
    "scripts/test_gemini.py",
]

for sf in stale_candidates:
    exists = (repo / sf).exists()
    if exists:
        tracked = run(f'git ls-files --error-unmatch "{sf}" 2>nul && echo TRACKED || echo UNTRACKED')
        is_tracked = "TRACKED" in tracked
        check(f"{sf} should not exist", False, f"EXISTS and tracked={is_tracked}")
    else:
        check(f"{sf} should not exist", True, "not present")

# Check for __pycache__ or .pyc in git
pyc_tracked = run('git ls-files "*.pyc" "__pycache__/*" 2>nul || echo NONE')
check("No .pyc/__pycache__ tracked", "NONE" in pyc_tracked or len(pyc_tracked.strip()) == 0,
      f"pyc tracked: {pyc_tracked.strip()[:100]}")

# Check for data/ and .joblib in git
data_tracked = run("git ls-files data/ 2>nul || echo NONE")
check("No data/ files tracked", "NONE" in data_tracked or len(data_tracked.strip()) == 0,
      f"data/ tracked: {data_tracked.strip()[:100]}")

joblib_tracked = run('git ls-files "*.joblib" 2>nul || echo NONE')
check("No .joblib files tracked", "NONE" in joblib_tracked or len(joblib_tracked.strip()) == 0,
      f"joblib tracked: {joblib_tracked.strip()[:100]}")

# Check for scratch/ tracked
scratch_tracked = run("git ls-files scratch/ 2>nul || echo NONE")
check("No scratch/ files tracked", "NONE" in scratch_tracked or len(scratch_tracked.strip()) == 0,
      f"scratch/ tracked: {scratch_tracked.strip()[:100]}")

# ============================================================
# 3. CODE CONSISTENCY: was the vacuous-truth fix applied to the REAL verifier?
# ============================================================
print("\n--- 3. Code consistency (vacuous-truth fix) ---")

verifier_src = (repo / "src" / "verifier.py").read_text()
# Check if verifier.py handles n_claims=0
has_zero_guard_v = "len(claims) == 0" in verifier_src or "n_claims == 0" in verifier_src or "len(per_claim) == 0" in verifier_src
check("verifier.py has vacuous-truth guard", has_zero_guard_v,
      f"guard in verifier.py: {has_zero_guard_v}")

pipeline_src = (repo / "src" / "pipeline.py").read_text()
has_zero_guard_p = "len(per_claim) == 0" in pipeline_src
check("pipeline.py has vacuous-truth guard", has_zero_guard_p,
      f"len(per_claim)==0 guard in pipeline.py: {has_zero_guard_p}")

# Check the specific lines
for i, line in enumerate(verifier_src.splitlines(), 1):
    if "faithfulness_score" in line and "claims" in line and "0" in line:
        print(f"    verifier.py:{i}: {line.strip()}")
for i, line in enumerate(pipeline_src.splitlines(), 1):
    if "len(per_claim) == 0" in line:
        print(f"    pipeline.py:{i}: {line.strip()}")
for i, line in enumerate((repo / "tests" / "test_verifier.py").read_text().splitlines(), 1):
    if "zero" in line.lower() or "claims" in line.lower():
        print(f"    test_verifier.py:{i}: {line.strip()}")

# Check which file the fix commit touched
fix_files = run("git show --name-only 67495b7 2>nul || echo CANNOT_SHOW_COMMIT")
print(f"    Fix commit changed: {fix_files.strip()[:200]}")

# ============================================================
# 4. OUTPUT CONSISTENCY: do the JSON files match the corrected numbers?
# ============================================================
print("\n--- 4. Output consistency ---")

summary_path = repo / "outputs" / "condition_comparison_summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    sg_mean = summary.get("sample_grounded", {}).get("mean", "?")
    co_mean = summary.get("class_only", {}).get("mean", "?")
    vo_mean = summary.get("values_only", {}).get("mean", "?")
    check("summary sample_grounded mean is ~0.96", abs(sg_mean - 0.96) < 0.02 if isinstance(sg_mean, (int, float)) else False,
          f"sample_grounded mean: {sg_mean}")
    check("summary class_only mean is ~0.34", abs(co_mean - 0.3377) < 0.02 if isinstance(co_mean, (int, float)) else False,
          f"class_only mean: {co_mean}")
    check("summary values_only mean is ~0.63", abs(vo_mean - 0.63) < 0.02 if isinstance(vo_mean, (int, float)) else False,
          f"values_only mean: {vo_mean}")
    vacuous_fixed = summary.get("vacuous_truth_cases_fixed", "?")
    print(f"    vacuous_truth_cases_fixed: {vacuous_fixed}")
else:
    check("condition_comparison_summary.json exists", False, "file missing")

# Check the 3 audit files have consistent indices
audit_files = [
    ("faithfulness_audit.json", "sample-grounded"),
    ("faithfulness_audit_classonly.json", "class-only"),
    ("faithfulness_audit_valuesonly.json", "values-only"),
]
all_indices = []
for fname, label in audit_files:
    fpath = repo / "outputs" / fname
    if fpath.exists():
        data = json.loads(fpath.read_text())
        indices = sorted([c["index"] for c in data])
        all_indices.append(indices)
        check(f"{label} has 100 cases", len(data) == 100, f"count={len(data)}")
    else:
        check(f"{label} exists", False, f"{fname} missing")
        all_indices.append([])

if len(all_indices) == 3 and all(len(x) == 100 for x in all_indices):
    check("All 3 audits use same 100 indices",
          all_indices[0] == all_indices[1] == all_indices[2],
          f"sg==co: {all_indices[0]==all_indices[1]}, sg==vo: {all_indices[0]==all_indices[2]}")

# Check faithfulness values in the audit JSONs
for fname, label, expected_mean in [
    ("faithfulness_audit.json", "sample-grounded", 0.96),
    ("faithfulness_audit_classonly.json", "class-only", 0.3377),
    ("faithfulness_audit_valuesonly.json", "values-only", 0.63),
]:
    fpath = repo / "outputs" / fname
    if fpath.exists() and len(all_indices[audit_files.index((fname, label))]) > 0:
        data = json.loads(fpath.read_text())
        actual_mean = sum(c["faithfulness"] for c in data) / len(data)
        check(f"{label} mean faithfulness matches expected",
              abs(actual_mean - expected_mean) < 0.02,
              f"actual={actual_mean:.4f}, expected={expected_mean}")

# Check the figure exists
fig_path = repo / "outputs" / "audit_condition_comparison.png"
if fig_path.exists():
    check("audit_condition_comparison.png exists", True, f"size={fig_path.stat().st_size} bytes")
else:
    check("audit_condition_comparison.png exists", False, "missing")

# ============================================================
# 5. REPRODUCIBILITY: README, requirements, .env.example, seeds
# ============================================================
print("\n--- 5. Reproducibility ---")

check("README.md exists", (repo / "README.md").exists(), "present")
check("requirements.txt exists", (repo / "requirements.txt").exists(), "present")
check(".env.example exists", (repo / ".env.example").exists(), "present")

if (repo / ".env.example").exists():
    env_example = (repo / ".env.example").read_text()
    mentions_nim = "nvidia" in env_example.lower() or "nim" in env_example.lower() or "integrate.api.nvidia.com" in env_example
    mentions_gemini = "google" in env_example.lower() or "gemini" in env_example.lower()
    check(".env.example references NIM (not Gemini)", mentions_nim, f"mentions NIM: {mentions_nim}")
    check(".env.example does not mention Gemini", not mentions_gemini, f"mentions Gemini: {mentions_gemini}")
    has_real_key = "nvapi-" in env_example
    check(".env.example has no real API key", not has_real_key, f"contains nvapi-: {has_real_key}")

# Check random_state=42
features_src = (repo / "src" / "features.py").read_text() if (repo / "src" / "features.py").exists() else ""
train_src = (repo / "src" / "train_baseline.py").read_text() if (repo / "src" / "train_baseline.py").exists() else ""
check("train_baseline uses random_state=42", "random_state=42" in train_src or "random_state=42" in features_src,
      f"random_state=42 in training code: {'random_state=42' in train_src}")

# ============================================================
# 6. FILE INVENTORY: what's actually in the repo?
# ============================================================
print("\n--- 6. File inventory ---")

all_tracked = run("git ls-files").strip().split("\n")
print(f"  Total tracked files: {len(all_tracked)}")

py_files = [f for f in all_tracked if f.endswith(".py")]
print(f"  Python files ({len(py_files)}):")
for f in sorted(py_files):
    print(f"    {f}")

json_files = [f for f in all_tracked if f.endswith(".json")]
print(f"  JSON files ({len(json_files)}):")
for f in sorted(json_files):
    print(f"    {f}")

md_files = [f for f in all_tracked if f.endswith(".md")]
print(f"  Markdown files ({len(md_files)}):")
for f in sorted(md_files):
    print(f"    {f}")

img_files = [f for f in all_tracked if f.endswith((".png", ".jpg", ".pdf"))]
print(f"  Image files ({len(img_files)}):")
for f in sorted(img_files):
    print(f"    {f}")

other = [f for f in all_tracked if not f.endswith((".py", ".json", ".md", ".png", ".jpg", ".pdf"))]
print(f"  Other files ({len(other)}):")
for f in sorted(other):
    print(f"    {f}")

# ============================================================
# 7. GIT LOG: recent commits
# ============================================================
print("\n--- 7. Recent git history ---")
log = run("git log --oneline -15")
print(log)

# ============================================================
# 8. TEST SUITE: run it
# ============================================================
print("\n--- 8. Test suite ---")
test_result = run("python -m pytest tests/ -v 2>&1")
print(test_result)

passed = test_result.count("PASSED")
failed = test_result.count("FAILED")
check(f"All tests pass ({passed} passed, {failed} failed)", failed == 0 and passed > 0,
      f"passed={passed}, failed={failed}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
n_pass = sum(1 for r in results if r[0] == "PASS")
n_fail = sum(1 for r in results if r[0] == "FAIL")
print(f"REPO AUDIT SUMMARY: {n_pass} PASS, {n_fail} FAIL out of {len(results)} checks")
if n_fail > 0:
    print("\nFAILED CHECKS:")
    for status, name, evidence in results:
        if status == "FAIL":
            print(f"  [FAIL] {name}: {evidence}")
print("=" * 80)