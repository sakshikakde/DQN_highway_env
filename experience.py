from common_utils import *

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)