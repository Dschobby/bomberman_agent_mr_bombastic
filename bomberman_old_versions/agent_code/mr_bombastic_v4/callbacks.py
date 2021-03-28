import os
import pickle
import random

import numpy as np


ACTIONS = ['RIGHT' , 'LEFT','UP', 'DOWN','WAIT','BOMB']


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
        #weights = np.random.rand(len(ACTIONS))
        #self.model = weights / weights.sum()
        self.model=np.zeros((len(ACTIONS),14))
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
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")


    #Here the agent should use the model 
    q_values=np.dot(self.model,state_to_features(game_state))
    best_action=ACTIONS[np.argmax(q_values)]

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

    #Agent coin Position
    agent_position=game_state['self'][3]
    if len(game_state['coins'])>0:
        agent_coin_position=np.array(game_state['coins'])-np.array(game_state['self'][3])
        distance=np.linalg.norm(agent_coin_position,axis=1)
        smallest_distance_index=np.argmin(distance)
        coin_direction_x=agent_coin_position[smallest_distance_index,0]
        coin_direction_y=agent_coin_position[smallest_distance_index,1]
        coin_right=np.sqrt(coin_direction_x**2+coin_direction_y**2)-np.sqrt((coin_direction_x-1)**2+coin_direction_y**2)
        coin_left=np.sqrt(coin_direction_x**2+coin_direction_y**2)-np.sqrt((coin_direction_x+1)**2+coin_direction_y**2)
        coin_up=np.sqrt(coin_direction_x**2+coin_direction_y**2)-np.sqrt(coin_direction_x**2+(coin_direction_y+1)**2)
        coin_down=np.sqrt(coin_direction_x**2+coin_direction_y**2)-np.sqrt(coin_direction_x**2+(coin_direction_y-1)**2)
    else: 
        coin_right=0
        coin_left=0
        coin_up=0
        coin_down=0
    #Sourrounding Blocks
    right_block=game_state['field'][agent_position[0]+1,agent_position[1]]
    left_block=game_state['field'][agent_position[0]-1,agent_position[1]]
    up_block=game_state['field'][agent_position[0],agent_position[1]-1]
    down_block=game_state['field'][agent_position[0],agent_position[1]+1]
    #Feature1: Reduced coin distance in each direction
    #Check if there is a wall
    if(right_block==-1):coin_right=-1
    if(left_block==-1):coin_left=-1
    if(up_block==-1):coin_up=-1
    if(down_block==-1):coin_down=-1




    #Feature2: Bomb activve
    bomb_active=int(game_state['self'][2])




    #Feature3: Reduced create distance in each direction
    if len(crates_positions(game_state))>0:
        agent_crate_position=crates_positions(game_state)-np.array(game_state['self'][3])
        distance=np.linalg.norm(agent_crate_position,axis=1)
        smallest_distance_index=np.argmin(distance)
        crate_direction_x=agent_crate_position[smallest_distance_index,0]
        crate_direction_y=agent_crate_position[smallest_distance_index,1]
    else:
        crate_direction_x=0
        crate_direction_y=0

    crate_right=np.sqrt(crate_direction_x**2+crate_direction_y**2)-np.sqrt((crate_direction_x-1)**2+crate_direction_y**2)
    crate_left=np.sqrt(crate_direction_x**2+crate_direction_y**2)-np.sqrt((crate_direction_x+1)**2+crate_direction_y**2)
    crate_up=np.sqrt(crate_direction_x**2+crate_direction_y**2)-np.sqrt(crate_direction_x**2+(crate_direction_y+1)**2)
    crate_down=np.sqrt(crate_direction_x**2+crate_direction_y**2)-np.sqrt(crate_direction_x**2+(crate_direction_y-1)**2)
    #Check if there is a wall
    if(right_block==-1):crate_right=-1
    if(left_block==-1):crate_left=-1
    if(up_block==-1):crate_up=-1
    if(down_block==-1):crate_down=-1




    #Feature4: Crate infront
    crate_infront=int(np.abs(crate_direction_x==1) or np.abs(crate_direction_y==1))




    #Feature5: Bomb distance
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
    else: 
        bomb_right=0
        bomb_left=0
        bomb_up=0
        bomb_down=0
    

    
    #Check if there is a wall
    if(right_block==-1):bomb_right=-1
    if(left_block==-1):bomb_left=-1
    if(up_block==-1):bomb_up=-1
    if(down_block==-1):bomb_down=-1



    #Feature5: Explosion map
    explosion_right=game_state['explosion_map'][agent_position[0]+1,agent_position[1]]
    explosion_left=game_state['explosion_map'][agent_position[0]-1,agent_position[1]]
    explosion_up=game_state['explosion_map'][agent_position[0],agent_position[1]-1]
    explosion_down=game_state['explosion_map'][agent_position[0],agent_position[1]+1]



    # Feature vector construction
    feature_vector=np.array([coin_right,coin_left,coin_up,coin_down,crate_right,crate_left,crate_up,crate_down,bomb_right,bomb_left,bomb_up,bomb_down,bomb_active,crate_infront])
    return feature_vector


def crates_positions(game_state: dict) -> np.array:
    result=np.where(game_state['field']==1)
    crate_array=np.array([result[0],result[1]]).T
    return crate_array
