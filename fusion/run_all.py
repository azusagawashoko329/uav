"""双卡动态调度全量训练 → 评估 → README(固定 seed,工程级完全重头)。

调度:场景队列 + 每张卡一个 worker,谁空谁取下一个(不让卡空着);每场景一个子进程跑 train_fusion
(两分支:外观梯度训练 + 行为从头重跑检测建点云,固定 --seed),stdout 落 logs/<scene>.log。
全部训练完成后跑 eval_fusion 出 7 场景融合表(logs/eval.log + results_fusion.csv),最后写 README.md。

产物:experiments/<YYYYMMDD>_all_model/{ckpt/<scene>/<scene>.{pth,npz}, det_cache/, logs/, README.md, results_fusion.csv}
Env: DMAD。例: python fusion/run_all.py --gpus 0 1 --seed 42
"""
import argparse, os, sys, subprocess, threading, queue, time, datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = ['crossroads', 'vehicle', 'bike', 'highway', 'farmland', 'solar', 'railway']  # 重→轻,利于双卡均衡


def worker(gpu, q, exp, logdir, env, results, epochs, seed):
    while True:
        try:
            sk = q.get_nowait()
        except queue.Empty:
            return
        log = os.path.join(logdir, f'{sk}.log'); t0 = time.time()
        print(f'[gpu{gpu}] ▶ 开始 {sk}', flush=True)
        with open(log, 'w') as f:
            rc = subprocess.run(
                [sys.executable, os.path.join(_ROOT, 'fusion', 'train_fusion.py'),
                 '--scenes', sk, '--gpu', str(gpu), '--exp', exp, '--epochs', str(epochs), '--seed', str(seed)],
                stdout=f, stderr=subprocess.STDOUT, env=env, cwd=_ROOT).returncode
        results[sk] = (gpu, time.time() - t0, rc)
        print(f'[gpu{gpu}] ✔ 完成 {sk} ({time.time()-t0:.0f}s, rc={rc})', flush=True)


def md_table(csv_path):
    if not os.path.exists(csv_path):
        return ['(eval 未产出 results_fusion.csv)']
    rows = [l.strip().split(',') for l in open(csv_path).read().splitlines()][1:]
    out = ['| 场景 | 外观 | 行为 | q | 融合 | Δ |', '|---|---|---|---|---|---|']
    A = Fz = 0.0
    for sk, aa, ab, q, af in rows:
        d = float(af) - float(aa); A += float(aa); Fz += float(af)
        out.append(f'| {sk} | {aa} | {ab} | {q} | **{af}** | {d:+.2f} |')
    n = len(rows)
    out.append(f'| **均值** | **{A/n:.2f}** | | | **{Fz/n:.2f}** | **{(Fz-A)/n:+.2f}** |')
    return out


def write_readme(exp, results, epochs, seed):
    base = os.path.join(_ROOT, 'experiments', exp)
    t = [f'# {exp} — 双分支统一训练(外观 RD + 行为 检测/描述子/kNN)', '',
         f'一个脚本、一个环境(DMAD: timm DINOv3 + ultralytics YOLO)同时产出两分支产物,**固定 seed={seed} 可复现**;'
         '行为分支的目标检测**从头重跑**(不读旧 `behavior/cache`,检测只缓存到本实验 `det_cache/`)。'
         '完整前向与融合见 [../../fusion/PIPELINE.md](../../fusion/PIPELINE.md)。', '',
         '## 训练配置(策略与历史一致)',
         f'- **外观**:DINOv3 ViT-B/16 frozen,taps=[5,7,9,11],OCBE code_dim=64,student depth=12,'
         f'top-k(k=5)重建损失,img=256,**{epochs} epoch**,bs=64,AdamW lr=5e-4 wd=5e-2 OneCycleLR,seed={seed}。',
         '- **行为**:YOLOv8x-VisDrone frozen 从头检测,5 维描述子 [d_min,d_mean,n_close,count,max_local],'
         'logz 标准化(计数维 log1p 后 z-score),合并 kNN-OOD(K=8),检测质量 q=min(1,0.1/jitter)。',
         '- **融合**:门控 `w=0.5·q·(1−2|p_app−0.5|)`,`fused=(1−w)·p_app+w·p_beh`,test 平均秩标准化,'
         '每场景 concat-micro AUC(MADViT 可比)。', '',
         '## 产物(按场景分目录,两分支同落 ckpt/<scene>/)',
         '- `ckpt/<scene>/<scene>.pth` — 外观 student+OCBE 权重 + cfg',
         '- `ckpt/<scene>/<scene>.npz` — 行为正常点云状态(Xtr_proc / mu / sd / q / jitter)',
         '- `det_cache/` — 本次从头检测的车辆中心点云  ·  `logs/<scene>.log` — 每场景训练日志  ·  `logs/eval.log`',
         '- `results_fusion.csv` — 评估结果', '',
         '## 结果(concat-micro AUC %)']
    t += md_table(os.path.join(base, 'results_fusion.csv'))
    if results:
        t += ['', '## 训练耗时', '| 场景 | GPU | 秒 | rc |', '|---|---|---|---|']
        t += [f'| {sk} | {g} | {s:.0f} | {rc} |' for sk, (g, s, rc) in results.items()]
    with open(os.path.join(base, 'README.md'), 'w') as f:
        f.write('\n'.join(t) + '\n')
    print(f'wrote {os.path.join(base, "README.md")}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    ap.add_argument('--exp', default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--scenes', default='all')
    a = ap.parse_args()
    exp = a.exp or datetime.date.today().strftime('%Y%m%d') + '_all_model'
    scenes = SCENES if a.scenes == 'all' else a.scenes.split(',')
    logdir = os.path.join(_ROOT, 'experiments', exp, 'logs'); os.makedirs(logdir, exist_ok=True)
    env = os.environ.copy(); env['HF_HUB_OFFLINE'] = '1'

    print(f'=== {exp} | GPUs={a.gpus} | {len(scenes)} 场景 | ep={a.epochs} | seed={a.seed} ===', flush=True)
    q = queue.Queue(); [q.put(s) for s in scenes]
    results = {}; T0 = time.time()
    threads = [threading.Thread(target=worker, args=(g, q, exp, logdir, env, results, a.epochs, a.seed))
               for g in a.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f'\n=== 训练完成,总耗时 {time.time()-T0:.0f}s,运行 eval_fusion ===', flush=True)

    elog = os.path.join(logdir, 'eval.log')
    with open(elog, 'w') as f:
        subprocess.run([sys.executable, os.path.join(_ROOT, 'fusion', 'eval_fusion.py'),
                        '--exp', exp, '--gpu', str(a.gpus[0])],
                       stdout=f, stderr=subprocess.STDOUT, env=env, cwd=_ROOT)
    print(open(elog).read(), flush=True)
    write_readme(exp, results, a.epochs, a.seed)
    print(f'\n=== 全部完成 → experiments/{exp}/ ===', flush=True)


if __name__ == '__main__':
    main()
