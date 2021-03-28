import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features
from .callbacks import crates_positions

import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 400  # 400 transitions(a full game) is rememberd
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Costum Events
COIN_DISTANCE_REDUCED_EVENT = "COIN_DISTANCE_REDUCED"
COIN_DISTANCE_INCREASED_EVENT = "COIN_DISTANCE_INCREASED"
CRATE_DISTANCE_REDUCED_EVENT = "CRATE_DISTANCE_REDUCED"
CRATE_DISTANCE_INCREASED_EVENT = "CRATE_DISTANCE_INCREASED"
BOMB_DISTANCE_REDUCED_EVENT = "BOMB_DISTANCE_REDUCED"
BOMB_DISTANCE_INCREASED_EVENT = "BOMB_DISTANCE_INCREASED"
IN_EXPLOSION_AREA_EVENT = "IN_EXPLOSION_AREA"
OUT_EXPLOSION_AREA_EVENT = "OUT_EXPLOSION_AREA"
BOMB_INFRONT_CRATE_EVENT = "BOMB_INFRONT_CRATE"
BOMB_BAD_PLACED_EVENT = "BOMB_BAD_PLACED"

# Actions
ACTIONS = ['RIGHT' , 'LEFT','UP', 'DOWN', 'WAIT', 'BOMB']
ACTIONS_DIC=dict(zip(ACTIONS,[0,1,2,3,4,5]))

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


    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    ################# Own rewards ######################################

    # If distance to coin is reduced get reward
    if old_game_state != None:
        if len(old_game_state['coins'])>0 and len(new_game_state['coins'])>0:
            coin_distances_old=np.linalg.norm(np.array(old_game_state['coins'])-np.array(old_game_state['self'][3]),axis=1)
            coin_distances_new=np.linalg.norm(np.array(new_game_state['coins'])-np.array(new_game_state['self'][3]),axis=1)
            samllest_distance_index=np.argmin(coin_distances_old)

            if len(coin_distances_new) == len(coin_distances_old):
                coin_distance_change = coin_distances_old[samllest_distance_index]-coin_distances_new[samllest_distance_index]
                if coin_distance_change > 0:
                    events.append(COIN_DISTANCE_REDUCED_EVENT)
                if coin_distance_change <= 0:
                    events.append(CRATE_DISTANCE_INCREASED_EVENT)

    #If distance to crate reduced get reward
    if old_game_state != None:
        if len(crates_positions(old_game_state))>0:
            crate_distances_old=np.linalg.norm(np.array(crates_positions(old_game_state)-np.array(old_game_state['self'][3])),axis=1)
            crate_distances_new=np.linalg.norm(np.array(crates_positions(new_game_state)-np.array(new_game_state['self'][3])),axis=1)
            samllest_distance_index=np.argmin(crate_distances_old)

            if len(crate_distances_new) == len(crate_distances_old):
                crate_distance_change = crate_distances_old[samllest_distance_index]-crate_distances_new[samllest_distance_index]
                if crate_distance_change > 0:
                    events.append(CRATE_DISTANCE_REDUCED_EVENT)
                if crate_distance_change <= 0:
                    events.append(CRATE_DISTANCE_INCREASED_EVENT)


    #if distance to bomb reduced get reward
    if old_game_state != None:
        if len(new_game_state['bombs'])>0 and len(old_game_state['bombs'])>0:
            bomb_distances_old=np.linalg.norm(np.array([np.array(old_game_state['bombs'][0][0])-np.array(old_game_state['self'][3])]),axis=1)
            bomb_distances_new=np.linalg.norm(np.array([np.array(new_game_state['bombs'][0][0])-np.array(new_game_state['self'][3])]),axis=1)
            samllest_distance_index=np.argmin(bomb_distances_old)

            if len(bomb_distances_new) == len(bomb_distances_old):
                bomb_distance_change = bomb_distances_old[samllest_distance_index]-bomb_distances_new[samllest_distance_index]
                if bomb_distance_change > 0:
                    events.append(BOMB_DISTANCE_REDUCED_EVENT)
                if bomb_distance_change < 0:
                    events.append(BOMB_DISTANCE_INCREASED_EVENT)


    #if get out of explosion area get reward
    if old_game_state != None:
        old_agent_position=old_game_state['self'][3]
        new_agent_position=new_game_state['self'][3]
        old_explosion_state=old_game_state['explosion_map'][old_agent_position[0],old_agent_position[1]]
        new_explosion_state=new_game_state['explosion_map'][new_agent_position[0],new_agent_position[1]]
        if old_explosion_state > 0 and new_explosion_state ==0:
            events.append(OUT_EXPLOSION_AREA_EVENT)
        if old_explosion_state > 0 or new_explosion_state >0:
            events.append(IN_EXPLOSION_AREA_EVENT)
    
    #if bomb gets placed infront of crate you get reward
    if old_game_state != None:
        crate_infront=False
        if len(crates_positions(new_game_state))>0:
            crate_distances=np.linalg.norm(np.array(crates_positions(new_game_state)-np.array(new_game_state['self'][3])),axis=1)
            if sum(crate_distances==1)>=1:
                crate_infront=True
        if crate_infront and e.BOMB_DROPPED in events:
            events.append(BOMB_INFRONT_CRATE_EVENT)
        else:
            events.append(BOMB_BAD_PLACED_EVENT)


    #####################################################################

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

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
        e.WAITED: -0.4,
        e.KILLED_SELF: -0.6,
        COIN_DISTANCE_REDUCED_EVENT: 0.2,
        COIN_DISTANCE_INCREASED_EVENT: -0.2,
        CRATE_DISTANCE_REDUCED_EVENT: 0.15,
        CRATE_DISTANCE_INCREASED_EVENT: -0.15,
        BOMB_DISTANCE_REDUCED_EVENT: -0.6,
        BOMB_DISTANCE_INCREASED_EVENT: 0.6,
        IN_EXPLOSION_AREA_EVENT: -0.2,
        OUT_EXPLOSION_AREA_EVENT: 0.2,
        BOMB_INFRONT_CRATE_EVENT: 0.8,
        BOMB_BAD_PLACED_EVENT: -0.5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
    