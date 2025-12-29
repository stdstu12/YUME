# bootstrap.py (simple)
import os, sys, runpy

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    # 注入路径：根目录、wan、wan23、hyvideo、fastvideo
    for sub in ("", "wan", "wan23", "hyvideo", "fastvideo"):
        p = os.path.join(ROOT, sub) if sub else ROOT
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    app_path = os.path.join(ROOT, "webapp_single_gpu.py")
    if not os.path.exists(app_path):
        raise FileNotFoundError(f"webapp_single_gpu.py not found at {app_path}")

    # 以 __main__ 方式执行原脚本，保留 from __future__ 位置
    runpy.run_path(app_path, run_name="__main__")

if __name__ == "__main__":
    main()
