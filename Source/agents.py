class Agent(object):
    def __init__(self):
        pass

    def train():
        pass

    def test():
        pass

    def forward():
        pass

    def backward():
        pass

class PokemonAgent(Agent):
    def __init__(self):
        super(PokemonAgent, self).__init__()

    def train(self):
        observation = self.env.getScreen()
        action = self.forward(observation)
        observation, reward = self.env.tick()
        self.backward(reward)
