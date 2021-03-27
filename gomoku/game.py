import numpy as np
from config.config import *

class Game():
    def __init__(self, board = np.zeros(WIDTH*LENGTH).reshape(LENGTH,WIDTH).astype(int)):
        self.board = board
        self.cache = None
    
    @property
    def action_space(self):
        if self.cache != None:
            return self.cache
        X,Y = np.where(self.board != 0)

        i,j = CENTER
        middle = list((i+di,j+dj) for di in range(-1,2) for dj in range(-1,2))
        
        if X.size == 0 and Y.size == 0:
            return middle

        placed = list(zip(X,Y))
        for i,j in placed:
            middle += self._nbr(i,j)
        return list((i,j) for i,j in set(middle) if self.board[i][j] == 0)
        
    @property
    def cur_player(self):
        if np.sum(self.board)%2:
            return WHITE
        else:
            return BLACK

    @property
    def is_ended(self):
        b = self.board
        for i in range(LENGTH):
            for j in range(WIDTH):
                if b[i][j] == 0:
                    continue
                # horizontal right
                if j+4 < WIDTH:
                    if b[i][j] == b[i][j+1] == b[i][j+2] == b[i][j+3] == b[i][j+4]:
                        return True
                    # left top right down
                    if i+4 < LENGTH and b[i][j] == b[i+1][j+1] == b[i+2][j+2] == b[i+3][j+3] == b[i+4][j+4]:
                        return True
                    # left down right top
                    if i >= 4 and b[i][j] == b[i-1][j+1] == b[i-2][j+2] == b[i-3][j+3] == b[i-4][j+4]:
                        if i==5 and j ==5:
                            print(3453455436546)
                        return True
                # vertical down
                if i+4 < LENGTH and b[i][j] == b[i+1][j] == b[i+2][j] == b[i+3][j] == b[i+4][j]:
                    return True
        return False

    @property
    def winner(self):
        if not self.is_ended:
            raise Exception("error: game is not ended")
        return BLACK if self.cur_player == WHITE else WHITE

    @property
    def random_action_idx(self):
        random_idx = np.random.choice(len(self.action_space))
        return self.action_space[random_idx]
    
    def take_action(self,i,j):
        if self.board[i][j] != 0:
            raise Exception(f"index ({i},{j}) already taken.")
        self.board[i][j] = self.cur_player
        ans = Game(board=np.copy(self.board))
        self.board[i][j] = 0
        return ans

    def _nbr(self,i,j):
        return [(i+b0*d,j+b1*d) for b0 in [-1,1] for b1 in [-1,1] for d in range(3) if (0<=i+b0*d<LENGTH and 0<=j+b1*d<WIDTH)]

    def display(self):
        print(''.join(["  "] + [str(i).rjust(2) for i in range(WIDTH)]))
        for i in range(LENGTH):
            line = [str(i).rjust(2)]
            for j,t in enumerate(self.board[i]):
                j = i*LENGTH + j
                char = O if t == WHITE else (X if t == BLACK else E)
                line.append(char)
            print(''.join(line))
        print("-"*100)

if __name__ == "__main__":
    game = Game()
    while not game.is_ended:
        i,j = game.random_action_idx
        game = game.take_action(i,j)
        game.display()