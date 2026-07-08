"""Quick environment check before running inference."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import os

os.chdir(ROOT)


def main() -> int:
    ok = True

    def check(label: str, fn):
        nonlocal ok
        try:
            fn()
            print(f"[OK] {label}")
        except Exception as exc:
            ok = False
            print(f"[FAIL] {label}: {exc}")

    def _py_version():
        if sys.version_info < (3, 10):
            raise AssertionError(f"need Python >= 3.10, got {sys.version.split()[0]}")

    check("Python >= 3.10", _py_version)
    check("PyTorch", lambda: __import__("torch"))
    check("CUDA", lambda: (_ for _ in ()).throw(RuntimeError("CUDA not available")) if not __import__("torch").cuda.is_available() else None)
    check("OpenCV", lambda: __import__("cv2"))
    def _gmod():
        try:
            __import__("gmod.gsplat")
        except ModuleNotFoundError:
            for base in (ROOT, ROOT.parent):
                if (base / "gmod").is_dir() and str(base) not in sys.path:
                    sys.path.append(str(base))
            __import__("gmod.gsplat")

    check("gmod renderer", _gmod)
    check("Fast2DGS models", lambda: __import__("models.GS_UNet"))
    check("Fast2DGS engine", lambda: __import__("engine"))
    check("heatmap weights", lambda: (_ for _ in ()).throw(FileNotFoundError("missing")) if not __import__("pathlib").Path("weights/smp_heat_div2k.pth").exists() else None)
    check("feat weights", lambda: (_ for _ in ()).throw(FileNotFoundError("missing")) if not __import__("pathlib").Path("weights/smp_feat_best_psnr_26_plus.pth").exists() else None)

    if ok:
        print("\nEnvironment ready. Run:")
        print("  python main_demo.py")
        print("  python inference.py --input assets/anime-1_2k.png")
        return 0
    print("\nFix the failures above, then re-run: python scripts/check_env.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())