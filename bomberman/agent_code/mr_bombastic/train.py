import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features
from .callbacks import target_positions
from .callbacks import bomb_positions
from .callbacks import sourrounding_blocks

import numpy as np

# Transition save
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
TRANSITION_HISTORY_SIZE = 400  # 400 transitions(a full game) is rememberd
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Costum Events
TARGET_DISTANCE_REDUCED_EVENT = "TARGET_DISTANCE_REDUCED"
TARGET_DISTANCE_INCREASED_EVENT = "TARGET_DISTANCE_INCREASED"
BOMB_DISTANCE_REDUCED_EVENT = "BOMB_DISTANCE_REDUCED"
BOMB_DISTANCE_INCREASED_EVENT = "BOMB_DISTANCE_INCREASED"
BOMB_INFRONT_TARGET_EVENT = "BOMB_INFRONT_TARGET"
BOMB_BAD_PLACED_EVENT = "BOMB_BAD_PLACED"
BOMB_NOT_POSSIBLE_EVENT = "BOMB_NOT_POSSIBLE"
NOT_IN_BOMB_AREA_EVENT = "NOT_IN_BOMB_AREA"

# Actions
ACTIONS = ['RIGHT' , 'LEFT','UP', 'DOWN', 'WAIT', 'BOMB']
ACTIONS_DIC=dict(zip(ACTIONS,[0,1,2,3,4,5]))

# Hyperparameters
learning_rate=0.1
discount_factor=0.5


def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    """
    Here our costum Events are defined
    """
    # Event if distance to target is increased or reduced
    if old_game_state != None:
        if len(target_positions(old_game_state))>0 and len(target_positions(new_game_state))>0:# and len(old_game_state['bombs'])==0:
            target_distances_old=np.linalg.norm(target_positions(old_game_state)-old_game_state['self'][3],axis=1)
            target_distances_new=np.linalg.norm(target_positions(new_game_state)-new_game_state['self'][3],axis=1)
            samllest_distance_index=np.argmin(target_distances_old)

            if len(target_distances_new) == len(target_distances_old):
                target_distance_change = target_distances_old[samllest_distance_index]-target_distances_new[samllest_distance_index]
                if target_distance_change > 0:
                    events.append(TARGET_DISTANCE_REDUCED_EVENT)
                if target_distance_change < 0:
                    events.append(TARGET_DISTANCE_INCREASED_EVENT)

    # Event if bomb distance is increased or reduced
    if old_game_state != None:
        if len(old_game_state['bombs'])>0 and np.min(np.linalg.norm(bomb_positions(old_game_state)-np.array(old_game_state['self'][3]),axis=1))<=4:
            bomb_distances_old=np.linalg.norm(bomb_positions(old_game_state)-np.array(old_game_state['self'][3]),axis=1)
            samllest_distance_index=np.argmin(bomb_distances_old)
            bomb_position=bomb_positions(old_game_state)[samllest_distance_index]
            bomb_distances_new=np.linalg.norm(bomb_position-np.array(new_game_state['self'][3]))

            bomb_distance_change = bomb_distances_old[samllest_distance_index]-bomb_distances_new
            if bomb_distance_change >= 0:
                events.append(BOMB_DISTANCE_REDUCED_EVENT)
            if bomb_distance_change < 0:
                events.append(BOMB_DISTANCE_INCREASED_EVENT)

    # Event if bomb is placed infront of crate
    if old_game_state != None:
        agent_position=new_game_state['self'][3]
        right_block,left_block,up_block,down_block=sourrounding_blocks(new_game_state)

        if (right_block==1 or left_block==1 or up_block==1 or down_block==1) and e.BOMB_DROPPED in events:
            events.append(BOMB_INFRONT_TARGET_EVENT)
        if (right_block!=1 and left_block!=1 and up_block!=1 and down_block!=1) and e.BOMB_DROPPED in events:
            events.append(BOMB_BAD_PLACED_EVENT)

    # Event if action BOMB was executed but not possible
    if old_game_state != None:
        if self_action =="BOMB" and not old_game_state['self'][2]:
            events.append(BOMB_NOT_POSSIBLE_EVENT)

    # Event if agent is in a bomb safe area
    if old_game_state != None:
        if len(new_game_state['bombs'])>0 and np.min(np.linalg.norm(bomb_positions(new_game_state)-np.array(new_game_state['self'][3]),axis=1))<=4:
            distance=np.min(np.linalg.norm(bomb_positions(new_game_state)-np.array(new_game_state['self'][3]),axis=1))
            if not old_game_state['self'][2] and distance%1!=0 and e.WAITED in events:
                events.append(NOT_IN_BOMB_AREA_EVENT)
        if len(old_game_state['bombs'])>0 and np.min(np.linalg.norm(bomb_positions(old_game_state)-np.array(new_game_state['self'][3]),axis=1))<=4:
            distance=np.min(np.linalg.norm(bomb_positions(old_game_state)-np.array(new_game_state['self'][3]),axis=1))
            if not old_game_state['self'][2] and distance%1!=0 and e.WAITED in events:
                events.append(NOT_IN_BOMB_AREA_EVENT)

    # State_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    """
    Here our model gets defined:
    q-learning with linear value function approximation
    """
    if(old_game_state)!=None:
        # Choosen action as index
        ac=ACTIONS_DIC[self_action]
        # Calculate Y
        q_values=np.dot(self.model,state_to_features(new_game_state))
        y=reward_from_events(self, events) + discount_factor * np.max(q_values)
        # Update the model
        self.model[ac]=self.model[ac] + learning_rate * state_to_features(old_game_state)*(y - np.dot(state_to_features(old_game_state),self.model[ac]))
    


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.
    This is also the place where the agent gets stored

    :param self: The same object that is passed to all of your callbacks.
    """
    # Save transitions
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    Here the rewards are defined
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.INVALID_ACTION: -2,
        TARGET_DISTANCE_REDUCED_EVENT: 1,
        TARGET_DISTANCE_INCREASED_EVENT: -1,
        BOMB_DISTANCE_REDUCED_EVENT: -2,
        BOMB_DISTANCE_INCREASED_EVENT: 2,
        BOMB_INFRONT_TARGET_EVENT: 1,
        BOMB_BAD_PLACED_EVENT: -1,
        BOMB_NOT_POSSIBLE_EVENT: -1,
        NOT_IN_BOMB_AREA_EVENT: 2
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
    