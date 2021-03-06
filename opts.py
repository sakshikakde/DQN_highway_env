import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument('--env', type=str, default='highway-v0', help='env name')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--gamma', default=0.999, type=float, help='Batch Size')

    parser.add_argument('--eps_start', default= 1, type=float, help='exploration rate start')
    parser.add_argument('--eps_end', default= 0.01, type=float, help='exploration rate end')
    parser.add_argument('--eps_decay', default= 0.001, type=float, help='exploration rate decay')

    parser.add_argument('--target_update', default=10, type=int, help='Interval after which target newtwork weights are updated')
    parser.add_argument('--memory_size', default=100000, type=int, help='replay memory size')

    parser.add_argument('--lr', default= 0.00025, type=float, help='Learning rate')
    parser.add_argument('--num_episodes', default=3000, type=int, help='Number of episodes')

    parser.add_argument('--save_interval', type=int, default=100, help='Episodes after which the model should be saved')
    parser.add_argument('--save_folder', type=str, default='/content/gdrive/MyDrive/ENPM690/Project/DQN_highway_env/snapshots', help='folder to save model')


    parser.add_argument('--model_name', type=str, default='highway-v0-Epoch-499-Duration-29.860000610351562_Apr-20-2022_0731.pth', help='Name of saved model')

    args = parser.parse_args()

    return args

