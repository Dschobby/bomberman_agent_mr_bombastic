import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features
from .callbacks import feature_to_index

import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 400  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
COIN_DISTANCE_REDUCED_EVENT = "COIN_DISTANCE_REDUCED"
COIN_DISTANCE_INCREASED_EVENT = "COIN_DISTANCE_INCREASED"
PLACEHOLDER_EVENT = "PLACEHOLDER"

#Hyperparameters
learning_rate = 0.2
discount_factor = 0.5

#Actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']
ACTIONS_DIC=dict(zip(ACTIONS,[0,1,2,3]))


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Coin Distance reward check
    if old_game_state != None:
        coin_distances_old=np.linalg.norm(np.array(old_game_state['coins'])-np.array(old_game_state['self'][3]),axis=1)
        coin_distances_new=np.linalg.norm(np.array(new_game_state['coins'])-np.array(new_game_state['self'][3]),axis=1)
        samllest_distance_index=np.argmin(coin_distances_old)

        if len(coin_distances_new) == len(coin_distances_old):
            coin_distance_change = coin_distances_old[samllest_distance_index]-coin_distances_new[samllest_distance_index]
            if coin_distance_change > 0.25:
                events.append(COIN_DISTANCE_REDUCED_EVENT)
            if coin_distance_change <= 0:
                events.append(COIN_DISTANCE_INCREASED_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    #Train the model: Q-learning
    if(old_game_state != None):
        action_index=ACTIONS_DIC[self_action]

        old_q_value = self.model[feature_to_index(state_to_features(old_game_state)),action_index]
        new_q_value = np.max(self.model[feature_to_index(state_to_features(new_game_state))])
        correction = learning_rate * (reward_from_events(self,events) + discount_factor * new_q_value - old_q_value)
        self.model[feature_to_index(state_to_features(old_game_state)),action_index] = old_q_value + correction


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    
    #print(np.count_nonzero(self.model==0))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -0.5,
        e.MOVED_RIGHT: -0.2,
        e.MOVED_LEFT: -0.2,
        e.MOVED_UP: -0.2,
        e.MOVED_DOWN: -0.2,
        COIN_DISTANCE_REDUCED_EVENT: 0.1,
        COIN_DISTANCE_INCREASED_EVENT: -0.12
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
