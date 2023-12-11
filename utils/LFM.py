import networkx as nx
import random
import matplotlib.pyplot as plt


class Community():
    ''' use set operation to optimize calculation '''

    def __init__(self, G, alpha=1.0):
        self._G = G
        self._alpha = alpha
        self._nodes = set()
        self._k_in = 0
        self._k_out = 0

    def add_node(self, node):
        neighbors = set(self._G.neighbors(node))
        # print("添加令居节点",neighbors , self._nodes,neighbors & self._nodes)
        node_k_in = len(neighbors & self._nodes)  # neighbor和self._nodes公有节点的数目存入node_k_in
        # print("node_k_in",node_k_in)
        node_k_out = len(neighbors) - node_k_in
        # print("node_k_out",node_k_out)
        self._nodes.add(node)
        self._k_in += 2 * node_k_in
        self._k_out = self._k_out + node_k_out - node_k_in

    def remove_node(self, node):
        neighbors = set(self._G.neighbors(node))
        community_nodes = self._nodes
        # print("community_nodes",community_nodes)
        node_k_in = len(neighbors & community_nodes)
        node_k_out = len(neighbors) - node_k_in
        self._nodes.remove(node)
        self._k_in -= 2 * node_k_in
        self._k_out = self._k_out - node_k_out + node_k_in

    def cal_add_fitness(self, node):  # fitness适应度
        neighbors = set(self._G.neighbors(node))
        old_k_in = self._k_in
        old_k_out = self._k_out
        vertex_k_in = len(neighbors & self._nodes)  # vertex顶点
        vertex_k_out = len(neighbors) - vertex_k_in
        new_k_in = old_k_in + 2 * vertex_k_in
        new_k_out = old_k_out + vertex_k_out - vertex_k_in
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self._alpha  # 幂次
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self._alpha
        return new_fitness - old_fitness

    def cal_remove_fitness(self, node):
        neighbors = set(self._G.neighbors(node))
        new_k_in = self._k_in
        new_k_out = self._k_out
        node_k_in = len(neighbors & self._nodes)
        node_k_out = len(neighbors) - node_k_in
        old_k_in = new_k_in - 2 * node_k_in
        old_k_out = new_k_out - node_k_out + node_k_in
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self._alpha
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self._alpha
        return new_fitness - old_fitness

    def recalculate(self):
        for vid in self._nodes:
            fitness = self.cal_remove_fitness(vid)
            if fitness < 0.0:
                return vid
        return None

    def get_neighbors(self):
        neighbors = set()
        for node in self._nodes:
            neighbors.update(set(self._G.neighbors(node)) - self._nodes)
        return neighbors

    def get_fitness(self):
        return float(self._k_in) / ((self._k_in + self._k_out) ** self._alpha)


class LFM():

    def __init__(self, G, alpha):
        self._G = G
        self._alpha = alpha

    def execute(self):
        communities = []
        print(list(self._G.nodes.keys()))
        print("---------------------")
        node_not_include = list(self._G.nodes.keys())
        while (len(node_not_include) != 0):
            c = Community(self._G, self._alpha)
            seed = random.choice(node_not_include)
            c.add_node(seed)
            print("随机选取节点是：", seed)
            to_be_examined = c.get_neighbors()
            print("c.get_neighbors()", c.get_neighbors())
            while (to_be_examined):
                # largest fitness to be added
                m = {}
                for node in to_be_examined:
                    fitness = c.cal_add_fitness(node)  # 计算点的适应度>0加入，小于0删除
                    m[node] = fitness
                """for m_item in m.items():
                    if m_item[1] > 0.0:
                        c.add_node(m_item[0])"""
                to_be_add = sorted(m.items(), key=lambda x: x[1], reverse=True)[0]
                # 适应度降序排列
                # stop condition
                if (to_be_add[1] < 0.0):
                    break
                c.add_node(to_be_add[0])
                to_be_remove = c.recalculate()
                while (to_be_remove != None):
                    c.remove_node(to_be_remove)
                    to_be_remove = c.recalculate()

                to_be_examined = c.get_neighbors()

            for node in c._nodes:
                if (node in node_not_include):
                    node_not_include.remove(node)


            communities.append(c._nodes)
        return communities


if __name__ == "__main__":
    # 创建一个图
    G = nx.Graph()
    # 添加边
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 3), (6, 7), (7, 8), (8, 6)])

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'label')
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1500)
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.axis('off')
    plt.show()

    # 创建LFM对象
    alpha_value = 0.8  # 设置alpha值
    lfm = LFM(G, alpha_value)

    # 执行社区发现
    communities = lfm.execute()

    # 打印结果
    print("Detected Communities:")
    for i, community_nodes in enumerate(communities):
        print(f"Community {i + 1}: {community_nodes}")