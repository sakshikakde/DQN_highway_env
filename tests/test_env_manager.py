import sys
from pathlib import Path
sys.path.append(str(Path( __file__ ).parent.joinpath('..')))
from env_manager import *

def test_reset(em):
    em.reset()
    assert(em.current_screen is None)

def test_num_actions(em):
    assert(em.num_actions_available() == em.env.action_space.n)

def test_init_data(em, device):
    screen = em.get_processed_screen().cpu().numpy()
    em.close()

    _, axes = plt.subplots(ncols=4, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(screen[0, i, ...].T, cmap=plt.get_cmap('gray'))
    plt.savefig('./tests/images/test_init.png')

def test_step_data(em, device):
    action = random.randrange(em.num_actions_available())
    em.take_action(torch.tensor([action]).to(device))
    screen = em.get_processed_screen().cpu().numpy()
    em.close()
    
    _, axes = plt.subplots(ncols=4, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(screen[0, i, ...].T, cmap=plt.get_cmap('gray'))
    plt.savefig('./tests/images/test_step.png')


if __name__ == "__main__":
    print("testing environment manager...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = HighwayEnvManager(device)
    test_reset(em)
    test_num_actions(em)
    test_init_data(em, device)
    test_step_data(em, device)