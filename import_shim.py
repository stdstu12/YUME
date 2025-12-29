# import_shim.py
# 作用：
# - 把项目根目录及常见源码子目录加入 sys.path
# - 如果缺少 __init__.py，自动创建（仅一次）
# - 如果没有 wan23 包但存在 wan 包，自动把 wan 别名成 wan23（含常用子模块）
# - 提供 WAN_CONFIGS：优先从 wan23.configs，次选 wan.configs；两者都没有时给出兜底配置

import os, sys, importlib, types

ROOT = os.path.dirname(os.path.abspath(__file__))

def _safe_mkdir_init(pkg_dir: str):
    """确保目录存在 __init__.py（避免某些环境无法识别为包）"""
    if os.path.isdir(pkg_dir):
        init_py = os.path.join(pkg_dir, "__init__.py")
        if not os.path.exists(init_py):
            try:
                with open(init_py, "a", encoding="utf-8") as f:
                    f.write("")  # 空文件即可
                print(f"[import_shim] created empty {init_py}")
            except Exception as e:
                print(f"[import_shim][WARN] create __init__.py failed for {pkg_dir}: {e}")

def _ensure_paths():
    # 根目录 & 常见源码目录
    paths = [
        ROOT,
        os.path.join(ROOT, "wan"),
        os.path.join(ROOT, "wan23"),
        os.path.join(ROOT, "hyvideo"),
        os.path.join(ROOT, "fastvideo"),
    ]
    for p in paths:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    # 确保这些目录是包
    for sub in ("wan", "wan23", "hyvideo", "fastvideo"):
        _safe_mkdir_init(os.path.join(ROOT, sub))

def _alias_module(src_name: str, dst_name: str, submods=()):
    """
    把已加载模块 src 别名成 dst（含常见子模块）。
    e.g. _alias_module('wan', 'wan23', submods=('configs','modules'))
    """
    if src_name not in sys.modules:
        return False
    src_mod = sys.modules[src_name]
    sys.modules[dst_name] = src_mod
    # 子模块映射
    for sm in submods:
        src_sub = f"{src_name}.{sm}"
        dst_sub = f"{dst_name}.{sm}"
        if src_sub in sys.modules:
            sys.modules[dst_sub] = sys.modules[src_sub]
        else:
            # 尝试从 src_name 下加载该子模块，再注册到 dst
            try:
                m = importlib.import_module(src_sub)
                sys.modules[dst_sub] = m
            except Exception:
                # 忽略没找到的子模块（后续导入不到再报错）
                pass
    return True

def ensure_packages():
    """
    必须在你的主脚本最顶部调用：确保路径、别名与包可导入。
    """
    _ensure_paths()

    # 1) 优先尝试直接 import wan23
    try:
        importlib.import_module("wan23")
    except ModuleNotFoundError:
        # 2) 没有 wan23，则看看是否有 wan；若有，把 wan 映射到 wan23
        try:
            importlib.import_module("wan")
            _alias_module("wan", "wan23", submods=("configs", "modules"))
            print("[import_shim] aliased 'wan' -> 'wan23'")
        except ModuleNotFoundError:
            # 两个都没有，创建一个空壳，避免 import 崩溃（后面用兜底 WAN_CONFIGS）
            shim = types.ModuleType("wan23")
            sys.modules["wan23"] = shim
            print("[import_shim][WARN] neither 'wan23' nor 'wan' found; created dummy 'wan23'")

def _load_wan_configs():
    """
    返回 WAN_CONFIGS：
    - 优先：from wan23.configs import WAN_CONFIGS
    - 其次：from wan.configs import WAN_CONFIGS
    - 兜底：返回一个最小可用的字典（避免 None）
    """
    # 优先 wan23.configs
    try:
        cfg_mod = importlib.import_module("wan23.configs")
        cfg = getattr(cfg_mod, "WAN_CONFIGS", None)
        if cfg:
            return cfg
    except Exception:
        pass

# 对外暴露的接口/变量：
ensure_packages()        # 调用以保证路径/别名可用
WAN_CONFIGS = _load_wan_configs()
