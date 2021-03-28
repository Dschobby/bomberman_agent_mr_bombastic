import os
import pickle
import random

import numpy as np


ACTIONS = ['RIGHT' , 'LEFT','UP', 'DOWN', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model=np.zeros((len(ACTIONS),9))
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        #print('random')
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")


    #Here the agent should use the model 
    q_values=np.dot(self.model,state_to_features(game_state))
    best_action=ACTIONS[np.argmax(q_values)]
    if(best_action=="BOMB"):
        #print('bomb')
        a=1
    return best_action


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None


    #Sourrounding Blocks
    agent_position=np.array(game_state['self'][3])
    right_block=game_state['field'][agent_position[0]+1,agent_position[1]]
    left_block=game_state['field'][agent_position[0]-1,agent_position[1]]
    up_block=game_state['field'][agent_position[0],agent_position[1]-1]
    down_block=game_state['field'][agent_position[0],agent_position[1]+1]


    #Feature1: Bomb activve
    bomb_active=int(game_state['self'][2])



    #Feature2: Crate around
    crate_infront=int(right_block==1 or left_block==1 or up_block==1 or down_block==1)

    bomb=bomb_active*crate_infront*2 -1



    #Feature3: Agent target position
    if len(target_positions(game_state))>0:
        target_position=target_positions(game_state)
        agent_target=target_position-agent_position
        distance=np.linalg.norm(agent_target,axis=1)
        smallest_distance_index=np.argmin(distance)

        target_direction_x=agent_target[smallest_distance_index,0]
        target_direction_y=agent_target[smallest_distance_index,1]
        target_right=np.sqrt(target_direction_x**2+target_direction_y**2)-np.sqrt((target_direction_x-1)**2+target_direction_y**2)
        target_left=np.sqrt(target_direction_x**2+target_direction_y**2)-np.sqrt((target_direction_x+1)**2+target_direction_y**2)
        target_up=np.sqrt(target_direction_x**2+target_direction_y**2)-np.sqrt(target_direction_x**2+(target_direction_y+1)**2)
        target_down=np.sqrt(target_direction_x**2+target_direction_y**2)-np.sqrt(target_direction_x**2+(target_direction_y-1)**2)

        #Check if there is a wall
        if(right_block!=0):target_right=-1
        if(left_block!=0):target_left=-1
        if(up_block!=0):target_up=-1
        if(down_block!=0):target_down=-1

        target_right=target_right*bomb_active
        target_left=target_left*bomb_active
        target_up=target_up*bomb_active
        target_down=target_down*bomb_active
    else:
        target_right=0
        target_left=0
        target_up=0
        target_down=0


    #Feature4: Bomb distance (should be zero if no bomb is there)
    if len(game_state['bombs'])>0:
        agent_bomb_position=np.array([np.array(game_state['bombs'][0][0])-np.array(game_state['self'][3])])
        distance=np.linalg.norm(agent_bomb_position,axis=1)
        smallest_distance_index=np.argmin(distance)
        bomb_direction_x=agent_bomb_position[smallest_distance_index,0]
        bomb_direction_y=agent_bomb_position[smallest_distance_index,1]
        bomb_right=-np.sqrt(bomb_direction_x**2+bomb_direction_y**2)+np.sqrt((bomb_direction_x-1)**2+bomb_direction_y**2)
        bomb_left=-np.sqrt(bomb_direction_x**2+bomb_direction_y**2)+np.sqrt((bomb_direction_x+1)**2+bomb_direction_y**2)
        bomb_up=-np.sqrt(bomb_direction_x**2+bomb_direction_y**2)+np.sqrt(bomb_direction_x**2+(bomb_direction_y+1)**2)
        bomb_down=-np.sqrt(bomb_direction_x**2+bomb_direction_y**2)+np.sqrt(bomb_direction_x**2+(bomb_direction_y-1)**2)
        #Check if there is a obstacal
        if(right_block!=0):bomb_right=-1
        if(left_block!=0):bomb_left=-1
        if(up_block!=0):bomb_up=-1
        if(down_block!=0):bomb_down=-1
    else: 
        bomb_right=0
        bomb_left=0
        bomb_up=0
        bomb_down=0


    # Feature vector construction
    feature_vector=np.array([target_right,target_left,target_up,target_down,bomb_right,bomb_left,bomb_up,bomb_down,bomb])
    return feature_vector


def target_positions(game_state: dict) -> np.array:
    result=np.where(game_state['field']==1)
    crate_array=np.array([result[0],result[1]]).T

    if len(crate_array)==0:
        target_array=np.array(game_state['coins'])
    if len(np.array(game_state['coins']))==0:
        target_array=crate_array
    if len(crate_array)!=0 and len(np.array(game_state['coins']))!=0:
        target_array=np.append(crate_array,np.array(game_state['coins']),axis=0)
    return np.array(target_array)
