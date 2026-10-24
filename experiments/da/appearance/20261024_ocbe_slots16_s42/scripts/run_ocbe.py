"""四组各自 16 维槽位，直接组成 64 维 code"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.run_dgt_variant import main
if __name__ == "__main__":
    sys.argv.extend(["--variant", "slot16", "--loss-mode", "group_balanced"])
    main()
