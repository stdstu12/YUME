"""
vbench_runner.py  --  Python replacement for vbench_runner.ps1
Calls sample_5b.py once per prompt with NUM_SAMPLES caption lines and a
fresh random seed, so the model loads once per prompt instead of once per sample.
"""
import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ALLOWED_TYPES = ['indoor', 'scenery']
NUM_SAMPLES   = 5
NUM_FRAMES    = 161


# ---------- helpers ----------

def _safe(s, maxlen=80):
    return re.sub(r'[<>:"/\\|?*]', '_', s)[:maxlen]

def _poll_vram(stop_event, readings):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL, text=True
            ).strip().splitlines()[0].strip()
            if out.isdigit():
                readings.append(int(out))
        except Exception:
            pass
        time.sleep(5)

def _ram_gb():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round(vm.used / (1024 ** 3), 2)
    except Exception:
        return ''

def _vram_peak_gb(readings):
    if not readings:
        return ''
    return round(max(readings) / 1024.0, 2)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vbench-json', required=True)
    ap.add_argument('--vbench-crop', required=True)
    ap.add_argument('--work-dir',    required=True)
    ap.add_argument('--num-frames',  type=int, default=NUM_FRAMES)
    args = ap.parse_args()

    work_dir   = Path(args.work_dir)
    out_base   = work_dir / 'outputs' / 'vbench' / 'videos'
    stats_path = work_dir / 'outputs' / 'vbench' / 'stats.csv'

    if not Path(args.vbench_json).exists():
        sys.exit(f'[vbench] ERROR: JSON not found: {args.vbench_json}')
    if not Path(args.vbench_crop).exists():
        sys.exit(f'[vbench] ERROR: crop dir not found: {args.vbench_crop}')

    with open(args.vbench_json, encoding='utf-8') as f:
        entries = json.load(f)

    seen, prompts = set(), []
    for e in entries:
        fn = e.get('file_name', '')
        if fn in seen:
            continue
        if e.get('type') not in ALLOWED_TYPES:
            continue
        seen.add(fn)
        cap = e.get('caption') or Path(fn).stem
        prompts.append({'file': fn, 'caption': cap, 'type': e['type']})

    total = len(prompts) * NUM_SAMPLES
    print(f'\n=== VBench prompt list ({len(prompts)} prompts, types: {", ".join(ALLOWED_TYPES)}) ===')
    for i, p in enumerate(prompts):
        img_ok = '  ' if (Path(args.vbench_crop) / p['file']).exists() else 'MISSING'
        print(f'  [{i+1:2}/{len(prompts)}] ({p["type"]:<8}) {img_ok} {p["caption"]}')
    print(f'=== {len(prompts)} prompts x {NUM_SAMPLES} samples = {total} runs ===\n')

    out_base.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    stats_is_new = not stats_path.exists()
    stats_f = open(stats_path, 'a', newline='', encoding='utf-8')
    writer  = csv.writer(stats_f)
    if stats_is_new:
        writer.writerow(['task_idx', 'prompt', 'type', 'sample_idx', 'seed',
                         'duration_s', 'gen_fps', 'vram_gb', 'ram_gb', 'out_path', 'status'])
        stats_f.flush()

    done = generated = errors = skipped = 0
    t_start = time.time()

    for ti, p in enumerate(prompts):
        img_src = Path(args.vbench_crop) / p['file']
        if not img_src.exists():
            print(f'[vbench] skip prompt {ti+1}: image not found - {img_src}')
            done += NUM_SAMPLES
            continue

        safe_cap   = _safe(p['caption'])
        safe_cap_u = safe_cap.replace(' ', '_')   # sample_5b replaces spaces → underscores

        def _cap_match(name):
            return safe_cap in name or safe_cap_u in name

        # samples already completed (renamed with _s{i}_seed{n} suffix)
        already = sorted([f for f in out_base.glob('*.mp4')
                          if _cap_match(f.name) and '_seed' in f.name])
        n_already = len(already)

        # orphans: files from an interrupted run that were never renamed
        orphans = sorted([f for f in out_base.glob('*.mp4')
                          if _cap_match(f.name) and '_seed' not in f.name])

        # recover orphans with new random seeds
        for i, src in enumerate(orphans):
            si_abs = n_already + i
            seed   = random.randint(0, 2**31 - 1)
            dst    = src.with_stem(f'{src.stem}_s{si_abs}_seed{seed}')
            print(f'  [recover] {src.name} -> {dst.name}')
            src.rename(dst)
            writer.writerow([ti, p['caption'], p['type'], si_abs, seed,
                              '', '', '', '', str(dst), 'recovered'])
            stats_f.flush()
            already.append(dst)
        n_already = len(already)

        n_needed = NUM_SAMPLES - n_already

        if n_needed <= 0:
            print(f'[vbench] skip prompt {ti+1}: all {NUM_SAMPLES} samples already done')
            for i, f in enumerate(already):
                writer.writerow([ti, p['caption'], p['type'], i, '', '', '', '', '', str(f), 'skipped'])
            stats_f.flush()
            skipped += NUM_SAMPLES
            done    += NUM_SAMPLES
            continue

        if n_already > 0:
            print(f'[vbench] prompt {ti+1}: {n_already} done, running {n_needed} remaining')

        short_cap = p['caption'][:60]

        # prepare temp dir with exactly this image (shared across samples)
        tmp_dir = work_dir / '_vbench_tmp'
        tmp_dir.mkdir(exist_ok=True)
        for f in tmp_dir.iterdir():
            f.unlink(missing_ok=True)
        shutil.copy(img_src, tmp_dir)

        cap_file = work_dir / '_vbench_caption.txt'
        cap_file.write_text(p['caption'], encoding='utf-8')

        for si in range(n_needed):
            si_abs = n_already + si
            seed   = random.randint(0, 2**31 - 1)

            pct = round(100 * done / total) if total else 0
            eta = ''
            if done > 0:
                elapsed = time.time() - t_start
                rem = int(elapsed / done * (total - done))
                eta = f'  ETA {rem//3600:02d}h{(rem%3600)//60:02d}m{rem%60:02d}s'
            print(f'[vbench] [{done+1}/{total}  {pct}%{eta}]  '
                  f'prompt {ti+1}/{len(prompts)} ({p["type"]})  '
                  f'sample {si_abs+1}/{NUM_SAMPLES}  seed {seed} : {short_cap}')

            vram_readings = []
            stop_evt    = threading.Event()
            vram_thread = threading.Thread(target=_poll_vram, args=(stop_evt, vram_readings), daemon=True)
            vram_thread.start()

            t0      = time.time()
            new_mp4 = None
            dur     = 0.0
            fps     = 0.0
            try:
                pre_existing = set(out_base.glob('*.mp4'))

                import socket as _socket
                with _socket.socket() as _s:
                    _s.bind(('127.0.0.1', 0))
                    _free_port = str(_s.getsockname()[1])
                _pythonpath = str(work_dir) + os.pathsep + os.environ.get('PYTHONPATH', '')
                env = {**os.environ,
                       'PYTHONPATH':             _pythonpath,
                       'TOKENIZERS_PARALLELISM': 'false',
                       'TF_ENABLE_ONEDNN_OPTS':  '0',
                       'LOCAL_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '1',
                       'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': _free_port}
                subprocess.run([
                    sys.executable, 'fastvideo/sample/sample_5b.py',
                    '--seed', str(seed),
                    '--gradient_checkpointing',
                    '--train_batch_size=1',
                    '--max_sample_steps=1',
                    '--mixed_precision=bf16',
                    '--allow_tf32',
                    '--t5_cpu',
                    f'--video_output_dir={out_base}',
                    f'--jpg_dir={tmp_dir}',
                    f'--caption_path={cap_file}',
                    '--test_data_dir=./val',
                    '--num_euler_timesteps', '5',
                    '--rand_num_img', '0.6',
                    '--internvl_path', './InternVL3-2B-Instruct',
                    '--height', '704',
                    '--width', '1216',
                    '--num_frames', str(args.num_frames),
                    '--fps', '24',
                ], cwd=str(work_dir), env=env)

                dur = round(time.time() - t0, 2)
                fps = round(args.num_frames / dur, 2)

                all_now  = set(out_base.glob('*.mp4'))
                raw_mp4s = sorted(
                    [f for f in (all_now - pre_existing)
                     if _cap_match(f.name) and '_seed' not in f.name],
                    key=lambda f: f.name
                )
                print(f'  [detect] {len(raw_mp4s)} new mp4s (pre={len(pre_existing)}, now={len(all_now)})')

                if raw_mp4s:
                    src = raw_mp4s[0]
                    dst = src.with_stem(f'{src.stem}_s{si_abs}_seed{seed}')
                    src.rename(dst)
                    new_mp4 = dst
                else:
                    print(f'  WARNING: expected 1 mp4, got 0')

            except Exception as exc:
                print(f'  EXCEPTION: {exc}', file=sys.stderr)
            finally:
                stop_evt.set()
                vram_thread.join(timeout=10)
                vram = _vram_peak_gb(vram_readings)
                ram  = _ram_gb()

            if new_mp4:
                print(f'  done in {dur}s  ({fps} gen-fps)  VRAM {vram}GB  RAM {ram}GB')
                writer.writerow([ti, p['caption'], p['type'], si_abs, seed,
                                  dur, fps, vram, ram, str(new_mp4), 'ok'])
                generated += 1
            else:
                writer.writerow([ti, p['caption'], p['type'], si_abs, seed,
                                  '', '', '', '', '', 'error'])
                errors += 1
                print(f'[vbench] ERROR prompt {ti+1} sample {si_abs} "{short_cap}" — continuing')
            stats_f.flush()
            done += 1

        shutil.rmtree(tmp_dir, ignore_errors=True)
        cap_file.unlink(missing_ok=True)

    stats_f.close()
    elapsed_m = round((time.time() - t_start) / 60, 1)
    print(f'\n[vbench] done - generated={generated}  skipped={skipped}  errors={errors}  elapsed={elapsed_m}m')
    print(f'[vbench] stats -> {stats_path}')


if __name__ == '__main__':
    main()
