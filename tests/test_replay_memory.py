import sys
from pathlib import Path
sys.path.append(str(Path( __file__ ).parent.joinpath('..')))
from replay_mem import *
from experience import *


def test_replay_mem_class():
    batch_size = 2
    rm = ReplayMemory(capacity = 4)
    assert(rm.can_provide_sample(batch_size) is False)

    rm.push(Experience(1, 2, 3, 4))
    rm.push(Experience(1, 3, 2, 4))
    rm.push(Experience(1, 4, 3, 2))
    rm.push(Experience(1, 2, 4, 3))

    assert(rm.can_provide_sample(batch_size) is True)

    sample = rm.sample(batch_size)
    assert(len(sample) == batch_size)

if __name__ == "__main__":
    print("testing replay memory class...")
    test_replay_mem_class()
