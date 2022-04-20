from common_utils import *

class HighwayEnvManager():
    def __init__(self, device):
        self.device = device
        self.env = gym.make('highway-v0').unwrapped
        self.configure()
        self.env.reset()
        self.current_screen = None
        self.observation = self.env.reset()
        self.done = False

    def configure(self):
        screen_width, screen_height = 200, 100
        config = {
            "observation": {
                "type": "GrayscaleObservation",
                "weights": [0.9, 0.1, 0.5],  # weights for RGB conversion
                "stack_size": 4,
                "observation_shape": (screen_width, screen_height)
            },
            "screen_width": screen_width,
            "screen_height": screen_height,
            "scaling": 5.75,
            "lanes_count":4,
        }
        self.env.configure(config)

    def reset(self):
        self.env.reset()
        self.observation = self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):        
        self.observation, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        self.current_screen = self.get_processed_screen()
        return self.current_screen

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_screen_stack(self):
        screen = self.get_processed_screen()
        return screen.shape[1]

    def get_processed_screen(self):
        screen = self.observation # PyTorch expects CHW
        # screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        # Strip off top and bottom
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):       
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # Use torchvision package to compose image transforms
        transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor()
        ])

        return transform(screen).unsqueeze(0).to(self.device) # add a batch dimension (BCHW)
