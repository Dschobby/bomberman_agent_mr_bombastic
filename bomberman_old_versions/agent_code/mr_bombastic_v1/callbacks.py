import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']


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
        self.model = np.zeros((256,4))
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

    q_values=self.model[feature_to_index(state_to_features(game_state))]
    best_action=ACTIONS[np.argmax(q_values)]
    #print(np.count_nonzero(self.model==0))
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


    # Feature construction
    #Coin direction (4 features left/right/... each 0 or 1)
    agent_position=game_state['self'][3]
    agent_coin_position=np.array(game_state['coins'])-np.array(game_state['self'][3])
    distance=np.linalg.norm(agent_coin_position,axis=1)
    smallest_distance_index=np.argmin(distance)
    coin_direction_x=agent_coin_position[smallest_distance_index,0]
    coin_direction_y=agent_coin_position[smallest_distance_index,1]

    right = int(coin_direction_x>0)
    left = int(coin_direction_x<0)
    up = int(coin_direction_y<0)
    down = int(coin_direction_y>0)
    #Sourrounding Blocks (4 features left/right/... each 0 or -1)
    right_block=game_state['field'][agent_position[0]+1,agent_position[1]]
    left_block=game_state['field'][agent_position[0]-1,agent_position[1]]
    up_block=game_state['field'][agent_position[0],agent_position[1]-1]
    down_block=game_state['field'][agent_position[0],agent_position[1]+1]

    #Coin Blocks (4 features each 0 or 1)


    # Feature vector construction
    feature_vector=np.array([right,left,up,down,right_block,left_block,up_block,down_block])
    # return feature vector
    return feature_vector

def feature_to_index(features: np.array) -> int:
    """
    This function assigns to each feature vector a specified index
    """
    #Possible values for each feature
    right = np.array([0,1])
    left = np.array([0,1])
    up = np.array([0,1])
    down =np.array([0,1])
    right_block = np.array([0,-1])
    left_block = np.array([0,-1])
    up_block = np.array([0,-1])
    down_block = np.array([0,-1])

    possible_features= np.array(np.meshgrid(right,left,up,down,right_block,left_block,up_block,down_block)).T.reshape(-1,8)

    return np.where((possible_features==features).all(1))[0][0]
