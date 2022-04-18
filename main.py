from common_utils import *

def main():
    opt = parse_opts()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = HighwayEnvManager(device)
    strategy = EpsilonGreedyStrategy(opt.eps_start, opt.eps_end, opt.eps_decay)
    agent = Agent(strategy, em.num_actions_available(), device)
    memory = ReplayMemory(opt.memory_size)\
    
    policy_net = DQN(em.get_screen_height(), em.get_screen_width(), em.get_screen_stack()).to(device)
    target_net = DQN(em.get_screen_height(), em.get_screen_width(), em.get_screen_stack()).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(params=policy_net.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    episode_durations = []
    for episode in range(opt.num_episodes):
        print("--------- Starting Episode : ", episode, "---------")
        duration = train_epoch(opt, em, agent, policy_net, target_net, memory, device, optimizer, criterion)
        print(duration)
        episode_durations.append(duration)
        # plot(episode_durations, 100)
        if episode % opt.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

if __name__ == "__main__":
	main()
