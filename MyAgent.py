#MyAgent.py# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
import numpy as np
from jass.agents.agent import Agent
from jass.game.const import PUSH, MAX_TRUMP, card_strings
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
import pandas as pd
import joblib
import os

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

class MYAgentMl(Agent):
    """
    Randomly select actions for the match of jass (Schieber)
    """
    def __init__(self):
        # log actions
        self._logger = logging.getLogger(__name__)
        # self._logger.setLevel(logging.INFO)
        # Use rule object to determine valid actions
        self._rule = RuleSchieber()
        # init random number generator
        self._rng = np.random.default_rng()
        self.model_clone = joblib.load('my_model.pkl')

    def action_trump(self, obs: GameObservation) -> int:
        """
        Select trump randomly. Pushing is selected with probability 0.5 if possible.
        Args:
            obs: the current match
        Returns:
            trump action
        """
        self._logger.info('Trump request')
        if obs.forehand == -1:
            # if forehand is not yet set, we are the forehand player and can select trump or push
            cards_tr = self._rule.get_valid_cards_from_obs(obs)
            cards_tr = np.append(cards_tr, 1).astype(bool)
            X_test_2 = np.transpose((pd.DataFrame(cards_tr)).values)
            X_test_3 = (pd.DataFrame(np.array([[True] * 37]))).values
        result = trump_c(np.array2string(self.model_clone.predict(X_test_2)))
        self._logger.info('Result: {}'.format(result))
        return result

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Select randomly a card from the valid cards
        Args:
            obs: The observation of the jass match for the current player
        Returns:
            card to play
        """
        self._logger.info('Card request')
        input_array=np.append(np.array(obs.hand), [np.array(obs.current_trick)/36, np.array(obs.trump)/6])
        # cards are one hot encoded
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        # convert to list and draw a value
        card = self._rng.choice(np.flatnonzero(valid_cards))
        self._logger.info('Played card: {}'.format(card_strings[card]))
        return card

