"""training-set logit margins (target logit minus log-sum-exp of the rest) at every checkpoint, to check for barron et al. 2026 softmax collapse (cliffs: 16.6 fp32, 36.7 fp64).

    uv run python scripts/check-margins.py RUN_DIR [RUN_DIR ...]
"""
import sys, torch as t
sys.path.insert(0, "src")
import analysis
dev = "cuda"
for rd in sys.argv[1:]:
    run = analysis.load_run(rd, device=dev)
    print("==", run["name"], "T", run["T"])
    for step, path in analysis.checkpoints(run):
        m = analysis.load_model(run, path, dev)
        ms = []
        with t.no_grad():
            for h, (x, y) in run["data"]["sets"]["train"].items():
                for i in range(0, len(x), 8192):
                    z = m(x[i:i+8192], run["T"], generator=analysis.eval_generator(run, dev))
                    z = z[0] if isinstance(z, tuple) else z
                    if z.dim() == 3: z = z[:, -1]
                    z = z.double(); yy = y[i:i+8192]
                    zt = z.gather(-1, yy[:, None])
                    oth = z.scatter(-1, yy[:, None], float("-inf"))
                    ms.append((zt[:, 0] - t.logsumexp(oth, -1)))
        mm = t.cat(ms)
        q = t.quantile(mm.float(), t.tensor([0.5, 0.99], device=dev))
        print(f"step {step:>6}  median {q[0]:6.1f}  p99 {q[1]:6.1f}  max {mm.max():6.1f}  frac>16.6 {(mm>16.6).float().mean():.3f}  frac>36.7 {(mm>36.7).float().mean():.3f}", flush=True)
