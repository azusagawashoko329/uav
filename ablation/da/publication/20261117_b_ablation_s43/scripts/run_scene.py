#!/usr/bin/env python3
from pathlib import Path
import sys
EXP=Path(__file__).resolve().parents[1]
ROOT=EXP.parents[3]
sys.path.insert(0,str(ROOT))
from scripts.da_publication_scene import main
if __name__ == '__main__':
    main(default_experiment=EXP)
