import copy
import random

import numpy as np


class Env:

    def __init__(self, nodes_path, adj_path, nfv_path):
        self.history = []
        self.__graph = Graph(nodes_path, adj_path, nfv_path)
        self.N = self.__graph.N
        self.__request = None
        self.__phase = None

        self.action_dim = 2 * self.N
        self.state_dim = 4 * self.N + self.N * self.N + 2 * self.__graph.k + 5

        self.__path = []  # 历史路径
        self.__cur_node = None  # 当前所在节点
        self.__cur_nfv_index = 0
        self.__cur_nfv = None  # 当前待部署的NFV
        self.__is_last_nfv = False

    def reset(self):
        """
        随机生成一批node，并且每次
        :return:
        """
        if len(self.history) > 100:
            self.render()
        self.__graph.init_nodes()
        self.__creat_sfc_request()  # 随机生成一个请求

        mask = self.__create_mask()
        if sum(mask) == 0:
            self.__creat_sfc_request()
            mask = self.__create_mask()

        return self.__create_state(), mask

    def step(self, action):

        """
        向前前进一步
        :param action: 要执行的动作，节点数的两倍，0-N表示选择该节点作为中间节点
        :return:
        """
        if isinstance(action, np.ndarray):
            action = action.item()
        node = action if action < self.N else action - self.N  # 计算选择哪个节点
        action_type = int(action / self.N)  # 将该节点作为转发节点还是部署节点，0 转发，1 部署

        if action_type == 0:
            self.__path.append((node,))
        elif action_type == 1 and not self.__is_last_nfv:
            self.__path.append((node, self.__cur_nfv_index))

            self.__graph.deploy_nfv(node, self.__cur_nfv_index)
            self.__cur_nfv_index += 1
            if self.__cur_nfv_index == len(self.__request.nfvs):
                self.__is_last_nfv = True
                self.__cur_nfv = None
            else:
                self.__cur_nfv = self.__graph.nfvs[self.__cur_nfv_index]

        self.__cur_node = node
        # 根据当前节点计算mask
        mask = self.__create_mask()

        reach = node == self.__request.dest and self.__is_last_nfv
        done = reach or sum(mask) == 0  # 如果部署完成
        reward = self.__cal_reward(reach)
        state = self.__create_state()
        return state, reward, bool(done), mask

    def render(self):
        with open("log/deploy.log", 'a') as file:
            for phase in self.history:
                log = f"{phase.start}, {phase.end}, {[nfv.name for nfv in phase.nfvs]}, {[tuple(p) for p in phase.path]} \n"
                file.write(log)

        self.history.clear()

    def __cal_reward(self, reach):
        """
        计算该步骤的奖励
        :param reach:
        :return:
        """
        reward = 500 if reach else -self.__graph.adj[self.__path[-1][0]][self.__cur_node]
        path_len = len(self.__path)
        reward += 50 * (self.__request.min_path_len - path_len)
        return reward

    def __create_mask(self):
        """
        根据当前节点计算动作的mask，1 不能走之前走过的节点 2 不能走资源不足的节点
        :return:
        """
        # 1 根据当前节点找到所有与当前节点相连的所有其余节点
        v = self.__graph.adjacency_vector(self.__cur_node)
        if not self.__is_last_nfv:  # 如果没有部署完成不能将目标节点作为可选节点
            v[self.__request.dest] = 0
        # 去除已经走过的节点
        for i in self.__path:
            v[i[0]] = 0

        # 2 如果相连节点的资源不足则不能作为部署节点
        if self.__is_last_nfv:
            v2 = [0 for _ in range(len(v))]
        else:
            v2 = [1 if self.__is_node_candidate(i) and a == 1 else a for i, a in enumerate(v)]

        v.extend(v2)

        return np.array(v)

    def __is_node_candidate(self, node_id):
        """
        判断给定节点是否能够部署当前的nfv
        :param node:
        :return:
        """
        if self.__is_last_nfv or node_id == self.__request.dest or node_id == self.__request.source:
            # 如果已经部署完成则不需要部署节点，目标节点和源节点不能作为部署节点
            return False

        node = self.__graph.nodes[node_id]
        nfv = self.__cur_nfv

        return node.cpus > nfv.cpus and node.mems > nfv.mems

    def __creat_sfc_request(self):
        """
        保存之前的请求的处理，并随机生成一个SFC请求
        :return:
        """

        self.__request = self.__graph.generate_sfc_request()

        if self.__phase is not None:
            self.__phase.path = self.__path
            self.history.append(self.__phase)
        self.__phase = Phase(self.__request.source, self.__request.dest, self.__request.nfvs)
        self.__path = [(self.__request.source,)]
        self.__cur_node = self.__request.source
        self.__cur_nfv_index = 0
        self.__cur_nfv = self.__request.nfvs[self.__cur_nfv_index]
        self.__is_last_nfv = False

    def __create_state(self):
        """
        生成状态，状态包含当前所有节点的状态，节点间的连接关系，
        :return:
        """
        state = []
        #  添加节点数据
        for node in self.__graph.nodes:
            state.append(node.id)
            state.append(node.cpus)
            state.append(node.mems)
            state.append(node.bws)

        # 添加邻接矩阵信息
        state.extend([element for sublist in self.__graph.adj for element in sublist])

        # 添加当前请求的信息
        state.append(self.__request.source)
        state.append(self.__request.dest)
        for nfv in self.__request.nfvs:
            state.append(nfv.id)
            state.append(nfv.delay)
        state.append(self.__request.delay)

        # 添加当前部署的信息
        state.append(self.__cur_node)
        state.append(self.__cur_nfv_index)

        return np.array(state, dtype=np.float64)


class SFCRequest:

    def __init__(self, source, dest, nfvs, delay, min_path_len):
        self.source = source
        self.dest = dest
        self.nfvs = nfvs
        self.delay = delay
        self.min_path_len = min_path_len


class NFV:

    def __init__(self, id, name, delay, cpus, mems):
        self.id = id
        self.name = name
        self.delay = delay
        self.cpus = cpus
        self.mems = mems


class Phase:

    def __init__(self, start, end, nfvs, path=[]):
        """
        :param start: 开始节点
        :param end: 结束节点
        :param path: 中间路径 [(node_id, nfv_id), (node_id), (node_id)]
        :param nfvs: 需要的nfv
        """

        self.start = start
        self.end = end
        self.path = path
        self.nfvs = nfvs


class Node:

    def __init__(self, id, resources):
        """

        :param id:
        :param resources: {cpus, mems, bws}
        """

        self.id = id
        self.cpus = resources['cpus']
        self.mems = resources['mems']
        self.bws = resources['bws']


class Graph:

    def __init__(self, nodes_path, adj_path, nfv_path, k=3):
        """
        使用字典保存每个节点的编号
        :param nodes_path:  节点路径
        :param adj_path: 对应的邻接关系路径
        :param nfv_path: nfv文件路径
        :param k: 生成的sfc请求中包含的nfv的最大数量
        """
        self.nodes_path = nodes_path
        self.adj_path = adj_path
        self.nfv_path = nfv_path
        self.source_nodes = None
        self.nodes = []
        self.init_nodes()

        self.N = len(self.nodes)
        self.adj = [[0 if i == j else 0 for j in range(self.N)] for i in range(self.N)]  # 初始化一个对角线为1的二维数组

        with open(adj_path, 'r') as f:
            for line in f.readlines():
                line = line.replace(" ", "")
                res = [int(i) for i in line.split(',')]
                self.adj[res[0]][res[1]] = res[2]
                self.adj[res[1]][res[0]] = res[2]

        self.nfvs = []
        with open(nfv_path, 'r') as f:
            for line in f.readlines():
                line = line.replace(" ", "")
                res = line.split(',')
                self.nfvs.append(NFV(int(res[0]), res[1], int(res[2]), int(res[3]), int(res[4])))
        self.nfv_nums = len(self.nfvs)
        self.k = k if self.nfv_nums > k else self.nfv_nums

    def init_nodes(self):
        self.nodes = []
        if self.source_nodes is None:
            self.source_nodes = []
            with open(self.nodes_path, 'r') as f:
                for line in f.readlines():
                    line = line.replace(" ", "")
                    res = [int(i) for i in line.split(',')]
                    self.source_nodes.append(Node(res[0], {
                        "cpus": res[1],
                        "mems": res[2],
                        "bws": res[3]
                    }))
        self.nodes = copy.deepcopy(self.source_nodes)

    def generate_sfc_request(self):
        """
        随机生成一个SFC请求
        :return:
        """

        source = random.randint(0, self.N - 1)
        dest = random.randint(0, self.N - 1)
        min_path_len = self.nth_power_linked(source, dest)
        while source == dest:
            dest = random.randint(0, self.N - 1)

        nfvs = [self.nfvs[i] for i in random.sample(range(self.nfv_nums), self.k)]
        # nfvs = [self.nfvs[0],self.nfvs[1],self.nfvs[2] ]

        # delay = 70
        delay = random.randint(60, 100)
        return SFCRequest(source, dest, nfvs, delay, min_path_len)

    def deploy_nfv(self, node_id, nfv_id):
        """

        :param node: 节点id
        :param nfv: nfv id
        :return: None
        """
        node = self.nodes[node_id]
        nfv = self.nfvs[nfv_id]
        node.cpus -= nfv.cpus
        node.mems -= nfv.mems

    def undeploy_nfv(self, node_id, nfv_id):
        """
        :param node: 节点id
        :param nfv: nfv id
        :return: None
        """
        node = self.nodes[node_id]
        nfv = self.nfvs[nfv_id]
        node.cpus += nfv.cpus
        node.mems += nfv.mems

    def adjacency_vector(self, node_id):
        """
        获得指定节点和其他节点的连接关系
        :param node_id:
        :return:
        """
        v = self.adj[node_id]
        return [1 if i > 0 else 0 for i in v]

    def nth_power_linked(self, source, dest):
        """
        根据矩阵的n次幂判断两个节点之间是否有链路相连，并返回最短路径长度
        :param source:
        :param dest:
        :return:
        """
        adj = np.array(self.adj)

        n = len(self.nfvs) + 1
        adj_power = np.linalg.matrix_power(adj, n)
        while adj_power[source][dest] <= 0 and n < 15:
            n += 1
            adj_power = np.linalg.matrix_power(adj, n)

        return n


if __name__ == '__main__':

    env = Env("src/graph/internet2.nodes.csv", "src/graph/internet2.adj.csv", "src/nfv/nfvs.csv")

    state, mask = env.reset()
    actions = []
    for i, e in enumerate(mask):
        if e > 0:
            actions.append(i)

    a = random.sample(actions, 1)[0]

    env.step(a)
