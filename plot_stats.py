from common_utils import *

def plot_stats(csv_file_name, opt):

    data = pd.read_csv(csv_file_name)
    if not (data.columns[0] == 'Duration'):  
        print("Adding headers")  
        header_list = ["Duration", "Reward", "Loss"]
        data = pd.read_csv(csv_file_name, names=header_list)
        
    duration = data['Duration']
    reward = data['Reward']
    loss = data['Loss']

    plot(duration, 50, opt, title = 'Training', y_label = 'Duration', name = 'plots/duration.png')
    plot(reward, 50, opt, title = 'Training', y_label = 'Rewards',  name = 'plots/reward.png')
    plot(loss, 50, opt, title = 'Training', y_label = 'Loss',  name = 'plots/loss.png')

def main():
    opt = parse_opts()
    csv_file_name  = os.path.join(opt.save_folder, opt.env, 'Apr-22-2022_0422_stats.csv')
    plot_stats(csv_file_name, opt)

if __name__ == "__main__":
	main()
