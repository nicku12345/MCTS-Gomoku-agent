import sys
sys.path.append('../')
import numpy as np
import os

import random
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt

from gomoku.game import Game

BLACK = 1
WHITE = -1
EMPTY = 0

class Layer1(nn.Module):
    def __init__(self):
        super(Layer1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2,out_channels=24,kernel_size=3)
        self.normalize1 = nn.LayerNorm([24,13,13])
        self.conv2 = nn.Conv2d(24,24*24,3)
        self.normalize2 = nn.LayerNorm([576,11,11])

    def forward(self,x):
        x = self.conv1(x)
        x = self.normalize1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.normalize2(x)
        x = F.relu(x)
        return x

class LayerP(nn.Module):
    def __init__(self):
        super(LayerP, self).__init__()
        self.conv1 = nn.Conv2d(576,576*2,1)
        self.normalize1 = nn.LayerNorm([1152,11,11])
        self.fc1 = nn.Linear(1152*11*11,15*15)

    def forward(self,x):
        x = self.conv1(x)
        x = self.normalize1(x)
        x = F.relu(x)
        x = x.view(x.size()[0],-1)
        x = self.fc1(x)
        return F.softmax(x, dim = -1)

class LayerV(nn.Module):
    def __init__(self):
        super(LayerV, self).__init__()
        self.conv = nn.Conv2d(576,576,1)
        self.normalize = nn.LayerNorm([576,11,11])
        self.fc1 = nn.Linear(69696,15*15)
        self.fc2 = nn.Linear(15*15,1)

    def forward(self,x):
        x = self.conv(x)
        x = self.normalize(x)
        x = F.relu(x)
        x = x.view(x.size()[0],-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.tanh(x)

class Model:
    def __init__(self,device = None):
        if not device:
            self.layer1 = Layer1()
            self.layerP = LayerP()
            self.layerV = LayerV()
        else:
            self.layer1 = Layer1().to(device)
            self.layerP = LayerP().to(device)
            self.layerV = LayerV().to(device)

        self.optimizer1 = optim.SGD(self.layer1.parameters(), lr = 0.1)
        self.optimizerP = optim.SGD(self.layerP.parameters(), lr = 0.1)
        self.optimizerV = optim.SGD(self.layerV.parameters(), lr = 0.1)

        self.lossP = nn.L1Loss()
        self.lossV = nn.L1Loss()

        self.batch_size = 10

    def save(self,dir=os.getcwd()):
        if 'checkpoint' not in os.listdir(dir):
            os.mkdir(dir+'/checkpoint/')
        dir += "/checkpoint/"
        name = datetime.now().strftime("/%m%d%H%M%S/")
        folder = dir+name
        os.mkdir(folder)
        torch.save(self.layer1.state_dict(),folder + "/layer1")
        torch.save(self.layerP.state_dict(),folder + "/layerP")
        torch.save(self.layerV.state_dict(),folder + "/layerV")

    def batch(self, memory, batch_size = None, num_batch = 1000):
        if not batch_size:
            batch_size = self.batch_size
        states = [] #[X,Y]
        search_prob = []
        values = []
        A = memory[:]
        np.random.shuffle(A)
        i = 0
        while i+batch_size < len(A):
            B = A[i:i+batch_size]
            S,P,V = [],[],[]
            for g,pi,v in B:
                S.append(self._g2XY(g))
                P.append(pi)
                V.append(v)

            states.append(S)
            search_prob.append(P)
            values.append(V)

            i += batch_size
        return states,search_prob,values

    def _g2XY(self,game):
        # game to [X,Y]
        board = game.board
        X = np.where(board == WHITE, 0, board)
        Y = np.where(board == BLACK, 0, board)
        Y = np.where(Y != EMPTY, 1, Y)
        return [X,Y]        

    def optimize(self, memory, batch_size = None):
        # perform gradient descent optimization using randomly batched data from memory
        if not batch_size:
            batch_size = self.batch_size
        states,search_prob,values = self.batch(memory)
        
        if torch.cuda.is_available():
            states = torch.FloatTensor(states).cuda()
            search_prob = torch.FloatTensor(search_prob).cuda()
            values = torch.FloatTensor(values).cuda()
        else:
            states = torch.FloatTensor(states)
            search_prob = torch.FloatTensor(search_prob)
            values = torch.FloatTensor(values)

        self.optimizer1.zero_grad()
        self.optimizerP.zero_grad()
        self.optimizerV.zero_grad()
        l = 0

        for XYs, pis, vs in zip(states,search_prob,values):
            vs = torch.reshape(vs,(1,batch_size))
            intermediate = self.layer1(XYs)

            P = self.layerP(intermediate)
            P = torch.reshape(P,(batch_size,15,15))

            V = self.layerV(intermediate)
            V = torch.reshape(V,(1,batch_size))
            print(P.shape, pis.long().shape)
            print(V.shape,vs.long().shape)
            l += self.lossP(P,pis.long()) + self.lossV(V,vs.long())

        l.backward()


    def parse_game(self, game):
        board = game.board
        idx = random.randint(0,4)
        for _ in range(idx):
            board = np.rot90(board)

        if random.randint(0,1):
            board = np.flip(board)

        # board consisting of only black
        X = np.where(board == WHITE, 0, board)

        # board consisting of only white
        Y = np.where(board == BLACK, 0, board)
        Y = np.where(Y != EMPTY, 1, Y)

        if torch.cuda.is_available():
            state = torch.cuda.FloatTensor([[X,Y]])
        else:
            state = torch.FloatTensor([[X,Y]])
        return state

    def evaluate_pv(self,game):
        state = self.parse_game(game)
        intermediate = self.layer1(state)
        P = self.layerP(intermediate)
        P = P.reshape(15,15)
        V = self.layerV(intermediate)
        return P.detach().cpu().numpy(), V.detach().cpu().numpy()[0][0]

    def evaluate_v(self,game):
        state = self.parse_game(game)
        intermediate = self.layer1(state)
        V = self.layerV(intermediate)
        return V.detach().cpu().numpy()[0][0]

    def heat_map(self, x):
        fig, ax = plt.subplots()
        im = ax.imshow(x)
        fig.tight_layout()
        cbar = ax.figure.colorbar(im, ax=ax)
        plt.show()


def main():
    print(f"Use cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device('cuda')

    game = Game()
    game = game.take_action(5,5)
    game = game.take_action(5,6)
    
    model = Model()
    P,V = model.evaluate_pv(game)
    print(P.shape)
    model.heat_map(P)
    print(V)
    

if __name__ == "__main__":
    main()