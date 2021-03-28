import os
import pickle
import random

import numpy as np


ACTIONS = ['RIGHT' , 'LEFT','UP', 'DOWN', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup the code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model=np.zeros((len(ACTIONS),10))
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    The agent parse the input, think, and take a decision.
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

    # The agent here uses the model to get the best action
    q_values=np.dot(self.model,state_to_features(game_state))
    best_action=ACTIONS[np.argmax(q_values)]
    return best_action


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # Sourrounding Blocks
    agent_position=np.array(game_state['self'][3])
    right_block,left_block,up_block,down_block=sourrounding_blocks(game_state)


    """
    Feature 1: is bomb active when crate or enemy infront
    Return: 0 or 1, 1 for yes 0 for no
    """
    # Is bomb active
    bomb_active=int(game_state['self'][2])
    # Is crate around
    crate_infront=int(right_block==1 or left_block==1 or up_block==1 or down_block==1)
    #Construction of the feature
    bomb=bomb_active*crate_infront*2 -1

    """
    Feature 2: is the agent out of the bombing area
    Return: 0 or 1, 1 for yes 0 for no
    """
    # Check if bombs are active and in a range of 4 steps to the agent
    if len(game_state['bombs'])>0 and np.min(np.linalg.norm(bomb_positions(game_state)-np.array(game_state['self'][3]),axis=1))<=4:
        agent_bomb_distance=np.min(np.linalg.norm(bomb_positions(game_state)-np.array(game_state['self'][3]),axis=1))
        not_in_bomb_area=int(agent_bomb_distance%1!=0)
    else:
        not_in_bomb_area=0


    """
    Feature 3: For each direction (right,left,up,down) check the distance reduction to the target if agent would move here.
               The target here is the nearest crate/agent/coin. 
               If the agent revealed a coin (coin range <4) this is the target
    Return: four numbers (for right/left/up/down)
    """
    if len(target_positions(game_state))>0:
        target_position=target_positions(game_state)
        agent_target=target_position-agent_position
        distance=np.linalg.norm(agent_target,axis=1)
        smallest_distance_index=np.argmin(distance)
        # Calculate target-agent relative coordinates
        target_direction_x=agent_target[smallest_distance_index,0]
        target_direction_y=agent_target[smallest_distance_index,1]
        # Calculate distance reduction
        target_right=np.sqrt(target_direction_x**2+target_direction_y**2)-np.sqrt((target_direction_x-1)**2+target_direction_y**2)
        target_left=np.sqrt(target_direction_x**2+target_direction_y**2)-np.sqrt((target_direction_x+1)**2+target_direction_y**2)
        target_up=np.sqrt(target_direction_x**2+target_direction_y**2)-np.sqrt(target_direction_x**2+(target_direction_y+1)**2)
        target_down=np.sqrt(target_direction_x**2+target_direction_y**2)-np.sqrt(target_direction_x**2+(target_direction_y-1)**2)
        # Check if there is a wall
        if(right_block!=0):target_right=-1
        if(left_block!=0):target_left=-1
        if(up_block!=0):target_up=-1
        if(down_block!=0):target_down=-1
        # If bomb is around do not go to target (run away from bomb)
        if len(game_state['bombs'])>0 and np.min(np.linalg.norm(bomb_positions(game_state)-np.array(game_state['self'][3]),axis=1))<=4:
            target_right=0
            target_left=0
            target_up=0
            target_down=0
    else:
        target_right=0
        target_left=0
        target_up=0
        target_down=0


    """
    Feature 4: For each direction (right,left,up,down) check the distance reduction to the bomb if agent would move here.
               The agent has to increase the distance
               If the agent has dropped a bomb he learns that he should first check in which direction there is free space to escape
    Return: four numbers (for right/left/up/down)
    """
    if len(game_state['bombs'])>0 and np.min(np.linalg.norm(bomb_positions(game_state)-np.array(game_state['self'][3]),axis=1))<=4:
        agent_bomb_position=bomb_positions(game_state)-np.array(game_state['self'][3])
        distance=np.linalg.norm(agent_bomb_position,axis=1)
        smallest_distance_index=np.argmin(distance)
        # Calculate bomb-agent relative coordinates
        bomb_direction_x=agent_bomb_position[smallest_distance_index,0]
        bomb_direction_y=agent_bomb_position[smallest_distance_index,1]
        # Calculate distance reduction
        bomb_right=-np.sqrt(bomb_direction_x**2+bomb_direction_y**2)+np.sqrt((bomb_direction_x-1)**2+bomb_direction_y**2)
        bomb_left=-np.sqrt(bomb_direction_x**2+bomb_direction_y**2)+np.sqrt((bomb_direction_x+1)**2+bomb_direction_y**2)
        bomb_up=-np.sqrt(bomb_direction_x**2+bomb_direction_y**2)+np.sqrt(bomb_direction_x**2+(bomb_direction_y+1)**2)
        bomb_down=-np.sqrt(bomb_direction_x**2+bomb_direction_y**2)+np.sqrt(bomb_direction_x**2+(bomb_direction_y-1)**2)
        # Go around edges is better whe escaping bomb -> smaller distance values get bigger
        if bomb_right>0: bomb_right=5/4-bomb_right  
        if bomb_left>0: bomb_left=5/4-bomb_left     
        if bomb_up>0: bomb_up=5/4-bomb_up   
        if bomb_down>0:bomb_down=5/4-bomb_down  
        # Check for free ways
        if np.linalg.norm(agent_bomb_position[smallest_distance_index])==0:
            if free_space(game_state)[0]==1: bomb_right=-1
            if free_space(game_state)[1]==1: bomb_left=-1
            if free_space(game_state)[2]==1: bomb_up=-1
            if free_space(game_state)[3]==1: bomb_down=-1
        # Check if there is a obstacal
        if(right_block!=0):bomb_right=-1
        if(left_block!=0):bomb_left=-1
        if(up_block!=0):bomb_up=-1
        if(down_block!=0):bomb_down=-1
        # Wait in safe area where bomb does not reach the agent
        bomb_right=bomb_right*(1-not_in_bomb_area)
        bomb_left=bomb_left*(1-not_in_bomb_area)
        bomb_up=bomb_up*(1-not_in_bomb_area)
        bomb_down=bomb_down*(1-not_in_bomb_area)
    else: 
        bomb_right=0
        bomb_left=0
        bomb_up=0
        bomb_down=0



    # Feature vector construction
    feature_vector=np.array([target_right,target_left,target_up,target_down,bomb_right,bomb_left,bomb_up,bomb_down,bomb,not_in_bomb_area])
    return feature_vector


def target_positions(game_state: dict) -> np.array:
    """
    This function transforms the gamestate in a numpy array with all possible 
    target coordinates the agent should go to

    :param game_state:  A dictionary describing the current game board.
    :return: numpy array (shape: N*2)
    """
    # Construct coordinate array of crates
    result=np.where(game_state['field']==1)
    crate_array=np.array([result[0],result[1]]).T
    # Construct coordinate array of enemy position
    enemy_array=np.zeros((len(game_state['others']),2))
    for i in range(len(game_state['others'])):
        enemy_array[i]=game_state['others'][i][3]
    # Check for empty arrays
    if len(crate_array)==0:
        target_array=np.array(game_state['coins'])
    if len(np.array(game_state['coins']))==0:
        target_array=crate_array
    if len(crate_array)!=0 and len(np.array(game_state['coins']))!=0:
        target_array=np.append(crate_array,np.array(game_state['coins']),axis=0)
    # Construct target array
    target_array=np.append(target_array,enemy_array,axis=0)
    # If coin near agent the target is the coin
    if len(np.array(game_state['coins']))>0 and np.min(np.linalg.norm(np.array(game_state['coins'])-np.array(game_state['self'][3]),axis=1))<=4.2:
        target_array=np.array(game_state['coins'])

    return np.array(target_array)

def bomb_positions(game_state: dict) -> np.array:
    """
    This function transforms the gamestate into a numpy array with all
    bomb coordinates

    :param game_state:  A dictionary describing the current game board.
    :return: numpy array (shape: N*2)
    """
    bomb_array=np.zeros((len(game_state['bombs']),2))
    for i in range(len(game_state['bombs'])):
        bomb_array[i]=np.array(game_state['bombs'][i][0])
    return bomb_array

def free_space(game_state: dict) -> np.array:
    """
    This function looks for a given gamestate in which direction there is free space
    where a bomb can not reach the agent
    1: free space avaiable
    0: no free space avilable

    :param game_state:  A dictionary describing the current game board.
    :return: numpy array (shape: 4)
    """
    # Atach new new boundarys so that you not access an index which not exist
    new_agent_position=np.array(game_state['self'][3])+np.array([1,1])
    new_game_field=np.insert(game_state['field'],0,-1,axis=1)
    new_game_field=np.insert(new_game_field,17,-1,axis=1)
    new_game_field=np.insert(new_game_field,0,-1,axis=0)
    new_game_field=np.insert(new_game_field,17,-1,axis=0)

    # Right free space
    right_1=np.abs(new_game_field[(new_agent_position+np.array([1,1]))[0],(new_agent_position+np.array([1,1]))[1]])
    right_2=np.abs(new_game_field[(new_agent_position+np.array([1,-1]))[0],(new_agent_position+np.array([1,-1]))[1]])
    right_3=1-(1 - np.abs(new_game_field[(new_agent_position+np.array([2,1]))[0],(new_agent_position+np.array([2,1]))[1]])) * (1 - np.abs(new_game_field[(new_agent_position+np.array([2,0]))[0],(new_agent_position+np.array([2,0]))[1]]))
    right_4=1-(1 - np.abs(new_game_field[(new_agent_position+np.array([2,-1]))[0],(new_agent_position+np.array([2,-1]))[1]])) * (1 - np.abs(new_game_field[(new_agent_position+np.array([2,0]))[0],(new_agent_position+np.array([2,0]))[1]]))
    # Left free space
    left_1=np.abs(new_game_field[(new_agent_position+np.array([-1,1]))[0],(new_agent_position+np.array([-1,1]))[1]])
    left_2=np.abs(new_game_field[(new_agent_position+np.array([-1,-1]))[0],(new_agent_position+np.array([-1,-1]))[1]])
    left_3=1-(1 - np.abs(new_game_field[(new_agent_position+np.array([-2,1]))[0],(new_agent_position+np.array([-2,1]))[1]])) * (1 - np.abs(new_game_field[(new_agent_position+np.array([-2,0]))[0],(new_agent_position+np.array([-2,0]))[1]]))
    left_4=1-(1 - np.abs(new_game_field[(new_agent_position+np.array([-2,-1]))[0],(new_agent_position+np.array([-2,-1]))[1]])) * (1 - np.abs(new_game_field[(new_agent_position+np.array([-2,0]))[0],(new_agent_position+np.array([-2,0]))[1]]))
    # Up free space
    up_1=np.abs(new_game_field[new_agent_position[0]+1,new_agent_position[1]-1])
    up_2=np.abs(new_game_field[new_agent_position[0]-1,new_agent_position[1]-1])
    up_3=1-(1 - np.abs(new_game_field[new_agent_position[0]+1,new_agent_position[1]-2])) * (1 - np.abs(new_game_field[new_agent_position[0],new_agent_position[1]-2]))
    up_4=1-(1 - np.abs(new_game_field[new_agent_position[0]-1,new_agent_position[1]-2])) * (1 - np.abs(new_game_field[new_agent_position[0],new_agent_position[1]-2]))
    # Down free space
    down_1=np.abs(new_game_field[new_agent_position[0]+1,new_agent_position[1]+1])
    down_2=np.abs(new_game_field[new_agent_position[0]-1,new_agent_position[1]+1])
    down_3=1-(1 - np.abs(new_game_field[new_agent_position[0]+1,new_agent_position[1]+2])) * (1 - np.abs(new_game_field[new_agent_position[0],new_agent_position[1]+2]))
    down_4=1-(1 - np.abs(new_game_field[new_agent_position[0]-1,new_agent_position[1]+2])) * (1 - np.abs(new_game_field[new_agent_position[0],new_agent_position[1]+2]))
    # Construct vector
    free_way=np.array([right_1*right_2*right_3*right_4,left_1*left_2*left_3*left_4,up_1*up_2*up_3*up_4,down_1*down_2*down_3*down_4])

    return free_way

def sourrounding_blocks(game_state: dict):
    """
    This function takes the current gamestate and returns a numpy array with the
    value for the sourrounding gameblocks
    -1: wall or bomb
     1: crate or agent

    :param game_state:  A dictionary describing the current game board.
    :return: numpy array (shape: 4)
    """
    agent_position=np.array(game_state['self'][3])
    new_game_field=game_state['field']  #np.copy..? so besser denke ich
    # Enemy blocks
    for i in range(len(game_state['others'])):
        x=game_state['others'][i][3][0]
        y=game_state['others'][i][3][1]
        new_game_field[x,y]=1
    # Bomb Blocks
    for i in range(len(game_state['bombs'])):
        x=int(bomb_positions(game_state)[i,0])
        y=int(bomb_positions(game_state)[i,1])
        new_game_field[x,y]=-1
    # Field blocks
    right_block=new_game_field[agent_position[0]+1,agent_position[1]]
    left_block=new_game_field[agent_position[0]-1,agent_position[1]]
    up_block=new_game_field[agent_position[0],agent_position[1]-1]
    down_block=new_game_field[agent_position[0],agent_position[1]+1]

    return right_block,left_block,up_block,down_block


#Possible additions
#relevant bombs: distance<=4 and choose therefore the bomb which explodes next
#look more far away for free space (if trained properly not necessaary)
#Bug!!!!!!!!! target values only 0 if own bomb is active !!
#Possible look for other bombs an if you go in explosion area of other bomb use -1/steps befor explosion -> writ function for this