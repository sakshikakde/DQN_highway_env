from common_utils import *
from models.dqn_conv_v1 import DQN as DQN
from train import *

def main():
    opt = parse_opts()
    print(opt)

    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)
    if not os.path.exists(os.path.join(opt.save_folder, opt.env)):
        os.mkdir(os.path.join(opt.save_folder, opt.env))

    timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
    csv_file_name  = os.path.join(opt.save_folder, f'{timestamp}_stats.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = HighwayEnvManager(device)
    strategy = EpsilonGreedyStrategy(opt.eps_start, opt.eps_end, opt.eps_decay)
    agent = Agent(strategy, em.num_actions_available(), device)
    memory = ReplayMemory(opt.memory_size)
    
    policy_net = DQN(em.get_screen_height(), em.get_screen_width(), em.get_screen_stack(), em.num_actions_available()).to(device)
    target_net = DQN(em.get_screen_height(), em.get_screen_width(), em.get_screen_stack(), em.num_actions_available()).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(params=policy_net.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    episode_durations = []
    episode_rewards = []
    best_reward = 0
    best_state = None

    for episode in range(opt.num_episodes):
        duration, reward, loss_epoch = train_epoch(opt, em, agent, policy_net, target_net, memory, device, optimizer, criterion)
        episode_durations.append(duration)
        episode_rewards.append(reward)

        write2csv(filename = csv_file_name, duration = duration, reward = reward, loss = loss_epoch)
        moving_avg_period = 50
        avg_reward = get_moving_average(moving_avg_period, episode_rewards)
        print("Episode", episode, "\n",
        moving_avg_period, "episode average reward: ", "{:.2f}".format(avg_reward[-1]), " | currect episode reward: ", "{:.2f}".format(reward), "| duration :", duration)

        state = {'epoch': episode, 'state_dict': policy_net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}

        if avg_reward[-1] > best_reward:
            best_reward = avg_reward[-1]
            best_state = state

        if episode % opt.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if episode % opt.save_interval == opt.save_interval - 1:
            timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
            torch.save(best_state, os.path.join(os.path.join(opt.save_folder, opt.env),
                                          f'{opt.env}-Epoch-{episode}-Duration-{ avg_reward[-1]}_{timestamp}.pth'))
            best_reward = 0
            print("Model saved with average duration ", avg_reward[-1])

if __name__ == "__main__":
	main()
