#MyAgent.py# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
import numpy as np
from jass.game.const import card_strings, card_values
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
        trump = 0
        max_number_in_color = 0
        for c in range(4):
            number_in_color = (obs.hand * card_values[c]).sum()
            if number_in_color > max_number_in_color:
                max_number_in_color = number_in_color
                trump = c
        return trump

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

