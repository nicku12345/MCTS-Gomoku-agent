import sys
sys.path.append('../')

import math

from gomoku.game import Game
from net.net import Model
from mct.cache import Cache
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

class Node:
    model = Model()
    def __init__(self, game = Game(), parent = None, par_idx = None, action_idx = None):
        self.game = game
        self.is_ended = self.game.is_ended

        self.action_idx = action_idx
        self.parent = parent

        # index of this node in parent.children
        self.par_idx = par_idx

        self.children = []
        self.children_total_score = None
        self.children_total_visit = None

        self.children_prior_prob_cache = None

        self.total_score = 0
        self.total_visit = 0

    def expand(self):
        if self.children:
            raise Exception("Node is already expanded")
        action_space = self.game.action_space

        idx = 0
        for i,j in action_space:
            game = self.game.take_action(i,j)
            self.children.append(Node(
                game = game,
                parent = self,
                par_idx = idx,
                action_idx = (i,j)))
            idx += 1

        self.children_total_score = np.zeros(idx)
        self.children_total_visit = np.zeros(idx)

    @property
    def children_prior_prob(self):
        if not self.children:
            raise Exception("error: node has no children")
        if self.children_prior_prob_cache:
            return self.children_prior_prob_cache
        ans = []
        
        P,V = self.model.evaluate_pv(self.game)
        for i,j in self.game.action_space:
            ans.append(P[i][j])
        return ans

    @property
    def child_Q(self):
        ans = self.children_total_score / self.children_total_visit
        ans[~np.isfinite(ans)] = float('inf')
        return ans

    @property
    def child_U(self):
        N = self.total_visit
        A = np.divide(self.children_prior_prob, 1 + self.children_total_visit)
        return min(math.log(1+N), 1.1) * A

    def iterate(self):
        if self.is_ended:
            return

        if not self.children:
            self.expand()

        me = self.game.cur_player

        idx = np.argmax(self.child_Q + self.child_U)
        cur = self.children[idx]

        while cur.children:
            idx = np.argmax(cur.child_Q + cur.child_U)
            cur = cur.children[idx]
        
        if not cur.is_ended:
            cur.expand()
            V = self.model.evaluate_v(cur.game)
        else:
            V = cur.game.winner

        V *= me           
        
        self.total_score += V
        self.total_visit += 1

        while cur.parent:
            par_idx = cur.par_idx
            cur.parent.children_total_visit[par_idx] += 1
            cur.parent.children_total_score[par_idx] += V

            cur.total_visit += 1
            cur.total_score += V

            cur = cur.parent

    def repeatedly_iterate(self,num_iter=500):
        for _ in range(num_iter):
            self.iterate()
            print(f"Step: {_}", end="\r", flush=True)

    @property
    def best_child_idx(self):
        return np.argmax(self.children_total_visit)

    @property
    def best_action_idx(self):
        idx = self.best_child_idx
        return self.game.action_space[idx]

def run_episode(cache):
    game = Game()
    while not game.is_ended:
        root = Node(game=game)
        root.repeatedly_iterate(num_iter=100)
        cache.save_state(root=root)
        i,j = root.best_action_idx
        game = game.take_action(i,j)
        game.display()
        print(cache.memory_size)
    cache.save_completed_game()

if __name__ == "__main__":
    cache = Cache()
    num_game = 0
    while num_game < 1:
        run_episode(cache=cache)
        num_game += 1
    Node.model.optimize(cache.memory)
    Node.model.save()