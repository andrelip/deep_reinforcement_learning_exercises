import gym
import collections
from torch.utils.tensorboard import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20
# import code
# code.interact(local=dict(globals(), **locals()))


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_random_step(self):
        action = self.env.action_space.sample()
        new_state, reward, is_done, _ = self.env.step(action)
        self.rewards[(self.state, action, new_state)] = reward
        self.transits[(self.state, action)][new_state] += 1
        self.state = self.env.reset() if is_done else new_state

    def calc_action_value(self, state, action):
        transits = self.transits[(state, action)]
        total_transitions = sum(transits.values())
        action_value = 0.0
        for target_state, target_count in transits.items():
            reward = self.rewards[(state, action, target_state)]
            action_value += (target_count / total_transitions) * \
                (reward + GAMMA * self.values[target_state])
        return action_value

    def calc_values(self):
        possible_states = self.env.observation_space.n
        for state in range(possible_states):
            possible_actions = self.env.action_space.n
            state_actions_values = [self.calc_action_value(
                state, action) for action in range(possible_actions)]
            max_action_values = max(state_actions_values)
            self.values[state] = max_action_values

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    # loop
    iter_no = 0
    best_reward = 0.0
    while True:
        for _ in range(100):
            agent.play_random_step()
        agent.calc_values()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.8:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
