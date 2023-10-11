from rl.agents.AgentBase import AgentBase

# DQN (off-policy)
from rl.agents.AgentDQN import AgentDQN, AgentDuelingDQN
from rl.agents.AgentDQN import AgentDoubleDQN, AgentD3QN

# off-policy
from rl.agents.AgentDDPG import AgentDDPG
from rl.agents.AgentTD3 import AgentTD3
from rl.agents.AgentSAC import AgentSAC, AgentModSAC

# on-policy
from rl.agents.AgentPPO import AgentPPO, AgentDiscretePPO
from rl.agents.AgentA2C import AgentA2C, AgentDiscreteA2C
