import torch
import random
import numpy as np
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

# number of items
MAX_MEMORY = 100_000

#
BATCH_SIZE = 1000

# Learning Rate
LR = 0.001


class Agent:
    def __init__(self) -> None:
        self.number_of_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate (must be smaller than 1)
        self.memory = deque(maxlen=MAX_MEMORY)  # popLeft when full
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        # danger area at one block in each direction
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight ahead
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.x,  # food up
            game.food.y > game.head.x,  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # pop left if MAX_MEMORY is reach
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE)  # returns list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # for one

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # There is some hard-coded vars to tweak later
    # More the epsilon is high, more random
    # Then it decrease with parties
    # then when it becomes negative, then full AI
    def get_action(self, state):
        # random moves: tradeoff exploration / exploration
        # If there is not move yet from the model, do random
        self.epsilon = 80 - self.number_of_games
        next_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            next_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            next_move[move] = 1

        return next_move


def train():
    plop_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state
        old_state = agent.get_state(game)

        # get move based on the current state
        final_move = agent.get_action(old_state)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(
            old_state, final_move, reward, new_state, done)

        # remember
        agent.remember(old_state, final_move, reward, new_state, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game:', agent.number_of_games,
                  'Score:', score, "Record:", record)
            
            plop_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_of_games
            plot_mean_scores.append(mean_score)

            plot(scores=plop_scores, mean_score=plot_mean_scores)


if __name__ == '__main__':
    train()
