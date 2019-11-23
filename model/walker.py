import itertools
import random
from joblib import Parallel, delayed
from tqdm import trange
from tqdm import tqdm

class HoraryWalker:
    def __init__(self, G, user_nodes):
        self.G = G
        self.user_nodes = set(user_nodes)
        self.nodes = list(self.G.nodes)
        self.t = dict.fromkeys(self.user_nodes, 0)

    def partition_num(self, num, workers):
        if num % workers == 0:
            return [num // workers] * workers
        else:
            return [num // workers] * workers + [num % workers]


    def horary_walk(self, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            curr = walk[-1]
            curr_nbrs = list(self.G.neighbors(curr))
            if len(curr_nbrs) > 0:
            # walk from user nodes using bias walk
                if curr in self.user_nodes:
                    ts_dict = {}
                    for curr_nbr in curr_nbrs:
                        if self.G[curr][curr_nbr]['weight'] > self.t[curr]:
                            ts_dict[curr_nbr] = self.G[curr][curr_nbr]['weight']
                    if ts_dict:
                        min_ts_nbr = min(ts_dict, key=ts_dict.get)
                        walk.append(min_ts_nbr)
                        self.t[curr] = ts_dict[min_ts_nbr]
                    else:
                        return walk

                # walk from user nodes using random walk
                else:
                    next_node = random.choice(curr_nbrs)
                    walk.append(next_node)
            else:
                return walk
        return walk

    def assign_walks(self, num_walks, walk_length, workers=1, verbose=0):
        print("sampling...")
        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(self.execute_walks)(num, walk_length) for num in
            self.partition_num(num_walks, workers))
        walks = list(itertools.chain(*results))
        return walks

    def execute_walks(self, num_walks, walk_length):
        walks = []
        for _ in trange(num_walks):
            random.shuffle(self.nodes)
            for node in tqdm(self.nodes):
                walks.append(self.horary_walk(
                    walk_length=walk_length, start_node=node))
        print("samping... done")

        return walks

