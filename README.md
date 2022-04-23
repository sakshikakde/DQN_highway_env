# DQN_highway_env
## Environment 
The ego-vehicle is driving on a multilane highway populated with other vehicles. The agent’s objective is to reach a high speed while avoiding collisions with neighboring vehicles. Driving on the right side of the road is also rewarded. I am using [OpenAI gym](https://gym.openai.com/). It is a toolkit for developing and comparing reinforcement learning algorithms.

### Installation
I am using gym version 0.21.1. Follow the instructions [here](https://highway-env.readthedocs.io/en/latest/installation.html) for the installation of highway environment.

### Action Space
For the lane changing task, I am using discrete meta action:      
0 : lane left       
1 : idle       
2 : lane right       
3 : faster       
4 : slower        
Refer the [doc](https://highway-env.readthedocs.io/en/latest/actions/index.html) for more details.

### Observation
The GrayscaleObservation is a W x H grayscale image of the scene, where W, H are set with the observation_shape parameter. The RGB to grayscale conversion is a weighted sum, configured by the weights parameter. Several images can be stacked with the stack_size parameter, as is customary with image observations. Refer the [doc](https://highway-env.readthedocs.io/en/latest/observations/index.html#grayscale-image) for more details.      

#### Sample obeservation


