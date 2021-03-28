import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features

import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 400  # 400 transitions(a full game) is rememberd
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Costum Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
COIN_DISTANCE_REDUCED_EVENT = "COIN_DISTANCE_REDUCED"
COIN_DISTANCE_INCREASED_EVENT = "COIN_DISTANCE_INCREASED"

# Actions
ACTIONS = ['RIGHT' , 'LEFT','UP', 'DOWN']
ACTIONS_DIC=dict(zip(ACTIONS,[0,1,2,3]))

# Hyperparameters
learning_rate=0.2
discount_factor=0.6

# Initialize variables
game_step = 0
model_correction = 0


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


    
    ##Possible the first step of a game should be ignored#############

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Total number of Gamesteps
    #game_step = game_step +1

    # If distance to coin is reduced get reward
    if old_game_state != None:
        coin_distances_old=np.linalg.norm(np.array(old_game_state['coins'])-np.array(old_game_state['self'][3]),axis=1)
        coin_distances_new=np.linalg.norm(np.array(new_game_state['coins'])-np.array(new_game_state['self'][3]),axis=1)
        samllest_distance_index=np.argmin(coin_distances_old)

        if len(coin_distances_new) == len(coin_distances_old):
            coin_distance_change = coin_distances_old[samllest_distance_index]-coin_distances_new[samllest_distance_index]
            if coin_distance_change > 0:
                events.append(COIN_DISTANCE_REDUCED_EVENT)
            if coin_distance_change <= 0:
                events.append(COIN_DISTANCE_INCREASED_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    # Model for the agent
    #if self_action!=None:
    #    ac=ACTIONS_DIC[self_action]
    #else: ac=4
    #batch_old_states=1
    #batch_new_states=1
    #batch_reward=1

    #for i in range(game_step):
    #    model_correction=learning_rate

    if(old_game_state)!=None:
        ac=ACTIONS_DIC[self_action]

        q_values=np.dot(self.model,state_to_features(new_game_state))
        y=reward_from_events(self, events) + discount_factor * np.max(q_values)

        self.model[ac]=self.model[ac] + learning_rate * state_to_features(old_game_state)*(y - np.dot(state_to_features(old_game_state),self.model[ac]))


    


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
        COIN_DISTANCE_REDUCED_EVENT: 0.2,
        COIN_DISTANCE_INCREASED_EVENT: -0.2
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def Q_function(self,game_state: dict, action: str):
    """
    This function calculates the actual Q-function according to the model
    """
    return np.dot(state_to_features(game_state),self.model[ACTIONS_DIC[action]])
    