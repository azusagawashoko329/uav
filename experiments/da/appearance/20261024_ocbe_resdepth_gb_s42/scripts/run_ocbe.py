"""当前 DGT 主路径加零初始化低秩深度变化残差，原始 loss"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.run_dgt_variant import main
if __name__ == "__main__":
    sys.argv.extend(["--variant", "residual_depth", "--loss-mode", "group_balanced"])
    main()
