"""Four independent depth Q matrices, independently randomized at seed 42."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.run_q_depthretain_group import main


if __name__ == "__main__":
    sys.argv.extend(["--basis-init", "random_orthonormal"])
    main()
