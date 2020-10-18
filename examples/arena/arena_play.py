# HSLU
#
# Created by Thomas Koller on 7/28/2020
#

import logging
import numpy as np

from jass.agents.agent import Agent
from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.game.const import color_masks, card_strings, card_values
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber

#from mcts.my_mcts_player import MyMCTSPlayer

#from mcts.my_mcts_player_random_trump import MyIMCTSPlayerRandomTrump
#from my_jass.player.my_player_deep_trump import MyPlayerDeepTrump
#from my_jass.ml_player.ml_player import MyMLPlayer
#from my_jass.imcts.my_imcts_deep_player import MyIMCTSDeepPlayer
#from my_jass.player.my_player import MyPlayer
import joblib
import pandas as pd
#from sklearn.model_selection import GridSearchCV



def trump_c(i):
    switcher = {
        "['DIAMONDS']": 0,
        "['HEARTS']": 1,
        "['SPADES']": 2,
        "['CLUBS']": 3,
        "['OBE_ABE']": 4,
        "['UNE_UFE']": 5,
    }
    return switcher.get(i)

class MyAgent2(Agent):
    """
    Sampl implemntation of a player to play Jass.
    """


    def __init__(self):
        # log actions
        self._logger = logging.getLogger(__name__)
        # Use rule object to determine valid actions
        self._rule = RuleSchieber()
        # init random number generator
        self._rng = np.random.default_rng()
        self.model_clone = joblib.load('my_model.pkl')

    def action_trump(self, obs: GameObservation) -> int:
        trump = 0
        max_number_in_color = 0
        # for c in range(4):
        #     number_in_color = (obs.hand * color_masks[c]).sum()
        #     if number_in_color > max_number_in_color:
        #         max_number_in_color = number_in_color
        #         trump = c
        cards_tr = self._rule.get_valid_cards_from_obs(obs)
        cards_tr =np.append(cards_tr,1).astype(bool)
        X_test_2 = np.transpose((pd.DataFrame(cards_tr)).values)
        X_test_3 = (pd.DataFrame(np.array([[True] * 37]))).values
        return trump_c(np.array2string(self.model_clone.predict(X_test_2)))

    def action_play_card(self, obs: GameObservation) -> int:
        # cards are one hot encoded
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        # convert to list and draw a value
        card = self._rng.choice(np.flatnonzero(valid_cards))
        self._logger.debug('Played card: {}'.format(card_strings[card]))
        return card


class MyAgent(Agent):
    """
    SW1 - Excercise
    """

    def __init__(self):
        # log actions
        self._logger = logging.getLogger(__name__)
        # Use rule object to determine valid actions
        self._rule = RuleSchieber()
        # init random number generator
        self._rng = np.random.default_rng()

    def action_trump(self, obs: GameObservation) -> int:
        trump = 0
        max_number_in_color = 0
        for c in range(4):
            number_in_color = (obs.hand * card_values[c]).sum()
            if number_in_color > max_number_in_color:
                max_number_in_color = number_in_color
                trump = c
        return trump

    def action_play_card(self, obs: GameObservation) -> int:
        # cards are one hot encoded
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        # convert to list and draw a value
        card = self._rng.choice(np.flatnonzero(valid_cards))
        self._logger.debug('Played card: {}'.format(card_strings[card]))
        return card


def main():
    # Set the global logging levl (Set to debug or info to see more messages)
    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(nr_games_to_play=1000, save_filename='arena_games')
    random_player = MyAgent()  # Team 1 # AgentRandomSchieber()
    my_player_1 = MyAgent2()    # Team 0 #billiher bot vom 1.Semester
    #mcts_player = MyMCTSPlayer()

    arena.set_players(my_player_1, random_player, my_player_1, random_player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
