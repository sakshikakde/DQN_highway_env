from common_utils import *

def plot_stats(csv_file_name, opt):
    data = pd.read_csv(csv_file_name)
    duration = data['Duration']
    reward = data['Reward']
    loss = data['Loss']

    plot(duration, 3, opt, title = 'Training', y_label = 'Duration', name = 'plots/duration.png')
    plot(reward, 3, opt, title = 'Training', y_label = 'Rewards',  name = 'plots/reward.png')
    plot(loss, 3, opt, title = 'Training', y_label = 'Loss',  name = 'plots/loss.png')

def main():
    opt = parse_opts()
    csv_file_name  = os.path.join(opt.save_folder, 'Apr-21-2022_1457_stats.csv')
    plot_stats(csv_file_name, opt)

if __name__ == "__main__":
	main()
