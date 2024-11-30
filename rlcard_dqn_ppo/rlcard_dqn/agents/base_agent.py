class BaseAgent:
    '''
    Custom base agent class to mimic RLCard's agent interface.
    '''

    def step(self, state):
        '''
        Select an action based on the current state.
        :param state: Current state from the environment
        :return: Action to take
        '''
        raise NotImplementedError("The step method must be implemented by the subclass")

    def feed(self, transition):
        '''
        Store a transition in the replay memory.
        :param transition: Tuple (state, action, reward, next_state, done)
        '''
        pass

    def eval_step(self, state):
        '''
        Evaluation step (optional). For evaluation purposes, without exploration.
        :param state: Current state from the environment
        :return: Chosen action (exploitation only)
        '''
        return self.step(state)