# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from rl.agents.AgentDQN import AgentDQN
from env.env import Env
from rl.train.config import Config
from rl.train.run import train_agent, train_agent_multiprocessing


def train():
    agent_class = AgentDQN
    env_class = Env
    env_args = {
        'env_name': "NFV-env",
        'num_envs': 1,
        'max_step': 100,
        'state_dim': 203,
        'action_dim': 24,
        'if_discrete': True,
        'nodes_path': "src/graph/internet2.nodes.csv",
        'adj_path': "src/graph/internet2.adj.csv",
        'nfv_path': "src/nfv/nfvs.csv"
    }

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(9e10)  # break training if 'total_step > break_step'
    args.net_dims = (128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.98  # discount factor of future rewards
    args.horizon_len = args.max_step * 2
    args.buffer_size = int(4e4)
    args.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** 0
    args.learning_rate = 1e-4

    args.eval_times = 32
    args.eval_per_step = 1e4

    args.gpu_id = GPU_ID
    args.num_workers = 4
    if_single_process = True
    if if_single_process:
        train_agent(args)
    else:
        train_agent_multiprocessing(args)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    GPU_ID = 0
    train()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
