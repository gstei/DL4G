# HSLU
#
# Created by Thomas Koller on 7/28/2020
#

import logging
import numpy as np

from jass.agents.agent import Agent
from jass.arena.arena import Arena
from jass.game.const import card_strings, card_values
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber

from jass.agents.agent_random_schieber import AgentRandomSchieber
from MyAgent import MYAgentMl

# from mcts.my_mcts_player import MyMCTSPlayera


class MyAgentRandom(Agent):

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
    random_player = AgentRandomSchieber()
    my_player_ml = MYAgentMl()

    arena.set_players(my_player_ml, random_player, my_player_ml, random_player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team MyPlayer: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points random_player: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
