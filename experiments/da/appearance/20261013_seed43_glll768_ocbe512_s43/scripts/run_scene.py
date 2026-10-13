"""Entrypoint for this experiment's scene runs; configuration is local."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.appearance_seed43_scene import main

if __name__ == "__main__":
    main()
