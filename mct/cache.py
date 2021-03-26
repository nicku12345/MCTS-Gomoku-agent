import numpy as np

class Cache:
    def __init__(self):
        self.memory = []
        self.current_games = []
        self.memory_size = 0

    def save_completed_game(self):
        end_game,v = self.current_games[-1]
        if not end_game.is_ended:
            winner = 0
        else:
            winner = end_game.winner

        for x in self.current_games:
            x.append(winner)
            self.memory.append(x)
            
        self.memory_size += len(self.current_games)
        self.current_games = []
        
    def save_state(self, root):
        # save [root.game, search_prob, .]
        search_prob = np.zeros(15*15).reshape(15,15)
        idx = 0
        for i,j in root.game.action_space:
            search_prob[i][j] += root.children_total_visit[idx]
            idx += 1
        search_prob /= root.total_visit
        self.current_games.append([root.game, search_prob])

    def reset(self):
        self.memory = []
        self.current_games = []

        