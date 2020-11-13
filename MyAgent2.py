#MyAgent2.py# HSLU
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
from ML_Agent import ML_Agent01
import pandas as pd
import joblib
import os
#Creating an Agent
agent = ML_Agent01(gamma=0.1, epsilon=1.0, batch_size=100, n_actions=36, eps_end=0.01, input_dims=[41], lr=0.1, eps_dec=5e-4, nn_size=64)
scores, eps_history = [], []
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

class MYAgentMl2(Agent):
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
        self.observation_new=[]
        self.tricks_played=0
        self.reward=0
        self.reward_history=[]
        self.card=0
        self.loss_history=[]

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

        observation_old=self.observation_new
        self.observation_new=np.append(np.append(np.array(obs.hand), np.array(obs.current_trick)/36), np.array(obs.trump)/6)
        self.observation_new=self.observation_new.astype(float)
        if(obs.trick_winner[self.tricks_played-1]==1):
            self.reward+=obs.trick_points[self.tricks_played-1]
        if(self.tricks_played>0):
            agent.store_transition(observation_old, self.card, self.reward, self.observation_new, False)
        aa = agent.learn()
        #print(aa)
        if aa is not None:
            self.loss_history.append(aa.cpu().data.numpy())
        self.tricks_played+=1
        if(self.tricks_played==9):
            self.tricks_played=0
            print('reward',self.reward, 'epsilon', agent.epsilon)
            self.reward_history=(self.reward_history).append(100)
            #if (len(self.reward_history)>101):
                #print(np.mean(self.reward_history[-100:]))
            self.reward=0
        self.card=agent.choose_action(self.observation_new)
        self._logger.info('Card request')
        # cards are one hot encoded
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        # convert to list and draw a value
        if not self.card in np.flatnonzero(valid_cards):
            self.card = self._rng.choice(np.flatnonzero(valid_cards))
            self.reward-=10

        self._logger.info('Played card: {}'.format(card_strings[self.card]))
        return self.card

