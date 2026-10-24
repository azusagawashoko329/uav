"""Review this checkpoint-enabled OCBE fusion experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.review_dgt_ckpt import main
if __name__ == '__main__':
    main()
