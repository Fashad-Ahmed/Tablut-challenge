from torch import device
from torch.cuda import is_available as cuda_is_available
from torch.backends.mps import is_available as mps_is_available

device = device(
    "cuda" if cuda_is_available() else
    "mps" if mps_is_available() else
    "cpu"
)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 81
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

WHITE_PAWNS = 9
BLACK_PAWNS = 16
NUM_MOVES = 81*81
NUM_OBSERVATIONS = 81
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 1000

IP_ADDRESS = "127.0.0.1"
WHITE_PORT = 5800
BLACK_PORT = 5801

SAVE_EVERY = 50

MEMORY_SIZE = 10000

WHITE_TENSOR_VALUE = 0.8
BLACK_TENSOR_VALUE = 0.3
EMPTY_TENSOR_VALUE = 0.0
KING_TENSOR_VALUE = 1.0