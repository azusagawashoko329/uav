import os
import sys
import runpy

if "--gpu" in sys.argv:
    sys.argv[sys.argv.index("--gpu") + 1] = "1"
runpy.run_path(os.path.join(os.path.dirname(__file__), "run_scene.py"), run_name="__main__")
