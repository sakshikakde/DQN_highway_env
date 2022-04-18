import sys
from pathlib import Path
sys.path.append(str(Path( __file__ ).parent.joinpath('..')))
from experience import *

def test_experience_class():
    exp = Experience(1, 2, 3, 4)
    print(exp)
    assert(exp.state == 1)
    assert(exp.action == 2)
    assert(exp.next_state == 3)
    assert(exp.reward == 4)

if __name__ == "__main__":
    print("testing experience class...")
    test_experience_class()
