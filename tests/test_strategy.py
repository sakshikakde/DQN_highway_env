import sys
from pathlib import Path
sys.path.append(str(Path( __file__ ).parent.joinpath('..')))
from strategy import *

def test_strategy_class():
    st = EpsilonGreedyStrategy(start = 10, end = 0.1, decay = 0.1)
    assert(int(st.get_exploration_rate(current_step = 5) - 6.1046)  == 0)

if __name__ == "__main__":
    print("testing strategy class...")
    test_strategy_class()