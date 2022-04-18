import sys
from pathlib import Path
sys.path.append(str(Path( __file__ ).parent.joinpath('..')))
from opts import *

def test_opts():
    opts = parse_opts()
    assert(opts.num_episodes == 1000)
    
if __name__ == "__main__":
    print("testing opt parser...")
    test_opts()