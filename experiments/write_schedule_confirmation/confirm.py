"""Run the prospectively selected schedule through the unchanged service."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'adaptive_reuse'))
import run_adaptive
if __name__=='__main__':
    run_adaptive.HERE=HERE
    run_adaptive.main()
