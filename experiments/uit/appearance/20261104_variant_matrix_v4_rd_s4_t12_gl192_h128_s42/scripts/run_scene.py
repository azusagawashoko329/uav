from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.uit_appearance_runner import main
if __name__ == "__main__":
    main()
