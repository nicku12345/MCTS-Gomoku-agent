from gomoku.game import Game
from mct.mct import Node
from mct.cache import Cache

def run_episode(cache, total = None, generation = 0):
    game = Game()
    while not game.is_ended:
        root = Node(game=game)
        root.repeatedly_iterate(num_iter=300)
        cache.save_state(root=root)
        i,j = root.best_action_idx
        game = game.take_action(i,j)
        game.display()
        print(f"Cache memory size: {cache.memory_size}")
        print(f"Total game: {total}")
    cache.save_completed_game()

if __name__ == "__main__":
    cache = Cache()
    total = 0
    generation = 0
    while True:
        num_game = 0
        while num_game < 1:
            run_episode(cache=cache, total = total, generation = generation)
            num_game += 1
            total += 1
        Node.model.optimize(cache.memory)
        Node.model.save()
        Node.model.reset()
        generation += 1