from common_utils import *
from models.dqn_conv_v1 import DQN as DQN


def load_model(model_path, model, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

def main():
    opt = parse_opts()
    model_path = os.path.join(opt.save_folder, opt.env, opt.model_name)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    em = HighwayEnvManager(device)
    strategy = EpsilonGreedyStrategy(0, 0, 0)
    agent = Agent(strategy, em.num_actions_available(), device)
    policy_net = DQN(em.get_screen_height(), em.get_screen_width(), em.get_screen_stack(), em.num_actions_available()).to(device)
    load_model(model_path, policy_net, device)
    print("Model loaded successfully")

    em.reset()
    state = em.get_state()

    for d in range(100):
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        state = next_state
        em.render(mode='rgb_array')
        if(em.done):
            print("Reward : ", reward, " | Duration : ", d)
            break

    em.close()
    
    
if __name__ == "__main__":
	main()