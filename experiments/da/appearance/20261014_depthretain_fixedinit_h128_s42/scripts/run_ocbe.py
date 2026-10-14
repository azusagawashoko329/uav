"""Fixed mean/trend starting Q for the depth-retention objective."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.run_q_depthretain import main


if __name__ == "__main__":
    sys.argv.extend(["--basis-init", "mean_trend"])
    main()
