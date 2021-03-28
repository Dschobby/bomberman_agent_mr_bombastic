import os
import pickle
import random

import numpy as np


ACTIONS = ['RIGHT' , 'LEFT','UP', 'DOWN']


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
        self.model=np.zeros((len(ACTIONS),8))
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
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25])

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

    agent_position=game_state['self'][3]
    agent_coin_position=np.array(game_state['coins'])-np.array(game_state['self'][3])
    distance=np.linalg.norm(agent_coin_position,axis=1)
    smallest_distance_index=np.argmin(distance)
    coin_direction_x=agent_coin_position[smallest_distance_index,0]
    coin_direction_y=agent_coin_position[smallest_distance_index,1]

    right_left = int(coin_direction_x>0) - int(coin_direction_x<0)
    up_down = int(coin_direction_y<0) - int(coin_direction_y>0)
    #if np.abs(coin_direction_x)>np.abs(coin_direction_y):
    #    right_left=right_left*2
    #if np.abs(coin_direction_x)<np.abs(coin_direction_y):
    #    up_down=up_down*2
    #inter_right_left_up_down=right_left*up_down
    #Sourrounding Blocks (4 features left/right/... each 0 or -1)
    right_block=game_state['field'][agent_position[0]+1,agent_position[1]]
    left_block=game_state['field'][agent_position[0]-1,agent_position[1]]
    up_block=game_state['field'][agent_position[0],agent_position[1]-1]
    down_block=game_state['field'][agent_position[0],agent_position[1]+1]
    #Interaction between sourrounding blocks
    #inter_right_left=right_block*left_block
    #inter_up_down=up_block*down_block
    #inter_right_up=right_block*up_block
    #inter_right_down=right_block*down_block
    #inter_left_up=left_block*up_block
    #inter_left_down=left_block*down_block
    field_situation=0
    if right_block==-1 and left_block==-1: field_situation=1
    if up_block==-1 and down_block==-1: field_situation=-1
    if right_block+up_block+down_block+left_block>=-1: field_situation=0


    #Coinblock..
    #if(coin_direction_x==1 and coin_direction_y==0): right_block=1
    #if(coin_direction_x==-1 and coin_direction_y==0): left_block=1
    #if(coin_direction_x==0 and coin_direction_y==-1): up_block=1
    #if(coin_direction_x==0 and coin_direction_y==1): down_block=1
    #Target coin in x and y direction free? -> feature
    coin_situation_x=0
    coin_situation_y=0
    coin_situation=0
    coin_position_x=game_state['coins'][smallest_distance_index][0]
    coin_position_y=game_state['coins'][smallest_distance_index][1]

    if game_state['field'][coin_position_x+1,coin_position_y]==-1 and game_state['field'][coin_position_x-1,coin_position_y]==-1: coin_situation_x=1
    if game_state['field'][coin_position_x+1,coin_position_y]==0 or game_state['field'][coin_position_x-1,coin_position_y]==0: coin_situation_x=0
    if game_state['field'][coin_position_x,coin_position_y+1]==-1 and game_state['field'][coin_position_x,coin_position_y-1]==-1: coin_situation_y=1
    if game_state['field'][coin_position_x,coin_position_y+1]==0 or game_state['field'][coin_position_x,coin_position_y-1]==0: coin_situation_y=0

    if coin_situation_x==0 and coin_situation_y==0: coin_situation=0
    if coin_situation_x==1 and coin_situation_y==0: coin_situation=1
    if coin_situation_x==0 and coin_situation_y==1: coin_situation=-1
    #inter_coin_situation=coin_situation_x*coin_situation_y
    inter_1=right_left*coin_situation
    inter_2=up_down*coin_situation
    inter_3=field_situation*coin_situation

    #ruduced coin distance

    coin_right=np.sqrt(coin_direction_x**2+coin_direction_y**2)-np.sqrt((coin_direction_x-1)**2+coin_direction_y**2)
    coin_left=np.sqrt(coin_direction_x**2+coin_direction_y**2)-np.sqrt((coin_direction_x+1)**2+coin_direction_y**2)
    coin_up=np.sqrt(coin_direction_x**2+coin_direction_y**2)-np.sqrt(coin_direction_x**2+(coin_direction_y+1)**2)
    coin_down=np.sqrt(coin_direction_x**2+coin_direction_y**2)-np.sqrt(coin_direction_x**2+(coin_direction_y-1)**2)
    if(right_block==-1):coin_right=0
    if(left_block==-1):coin_left=0
    if(up_block==-1):coin_up=0
    if(down_block==-1):coin_down=0

    # Feature vector construction
    feature_vector=np.array([coin_right,coin_left,coin_up,coin_down,right_block,left_block,up_block,down_block])
    return feature_vector
