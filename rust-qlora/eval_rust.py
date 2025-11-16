import os, random, subprocess, tempfile, json, jsonlines

def compile_and_clippy(snippets, sample_n=16):
    random.seed(0)
    picks = random.sample(snippets, min(sample_n, len(snippets)))
    ok_compile = 0
    clippy_warns = 0

    for code in picks:
        with tempfile.TemporaryDirectory() as td:
            # Minimal cargo project
            subprocess.run(["cargo","new","--quiet","app"], cwd=td, check=True)
            proj = os.path.join(td, "app")
            with open(os.path.join(proj, "src", "main.rs"), "w") as f:
                f.write(code)
            # Compile
            c1 = subprocess.run(["cargo","check","-q"], cwd=proj)
            ok_compile += int(c1.returncode == 0)
            # Clippy warnings count
            c2 = subprocess.run(["cargo","clippy","-q"], cwd=proj, capture_output=True, text=True)
            clippy_warns += c2.stdout.count(": warning:")
    return {"compile_rate": ok_compile/len(picks), "avg_clippy_warnings": clippy_warns/len(picks)}

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    samples = []
    with jsonlines.open(path) as r:
        for rec in r:
            samples.append(rec["gen"])
    metrics = compile_and_clippy(samples)
    print(json.dumps(metrics, indent=2))
