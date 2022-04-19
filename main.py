from common_utils import *
from train import *

def main():
    opt = parse_opts()
    print(opt)

    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)
    if not os.path.exists(os.path.join(opt.save_folder, opt.env)):
        os.mkdir(os.path.join(opt.save_folder, opt.env))

    timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
    f = open(os.path.join(opt.save_folder, f'{timestamp}_mv.csv'), 'w')
    writer = csv.writer(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = HighwayEnvManager(device)
    strategy = EpsilonGreedyStrategy(opt.eps_start, opt.eps_end, opt.eps_decay)
    agent = Agent(strategy, em.num_actions_available(), device)
    memory = ReplayMemory(opt.memory_size)\
    
    policy_net = DQN(em.get_screen_height(), em.get_screen_width(), em.get_screen_stack(), em.num_actions_available()).to(device)
    target_net = DQN(em.get_screen_height(), em.get_screen_width(), em.get_screen_stack(), em.num_actions_available()).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(params=policy_net.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    episode_durations = []
    for episode in range(opt.num_episodes):
        duration = train_epoch(opt, em, agent, policy_net, target_net, memory, device, optimizer, criterion)
        episode_durations.append(duration)

        moving_avg_period = 10
        moving_avg = get_moving_average(moving_avg_period, episode_durations)
        print("Episode", episode, "\n",
        moving_avg_period, "episode moving avg:", moving_avg[-1])

        writer.writerow([moving_avg[-1]])
        # # plot(episode_durations, 100)
        if episode % opt.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if episode % opt.save_interval == opt.save_interval - 1:
            state = {'epoch': episode, 'state_dict': policy_net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
            timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
            torch.save(state, os.path.join(os.path.join(opt.save_folder, opt.env),
                                          f'{opt.env}-Epoch-{episode}-Duration-{ moving_avg[-1]}_{timestamp}.pth'))
            print("Model saved with averahe duration ", moving_avg[-1])

    f.close()
    f = open(os.path.join(opt.save_folder, f'{timestamp}_duration.csv'), 'w')
    writer = csv.writer(f)
    writer.writerow(episode_durations)
    f.close()

if __name__ == "__main__":
	main()
