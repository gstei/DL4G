# HSLU
#
# Created by Thomas Koller on 7/28/2020
#

import logging
import numpy as np
import operator

from jass.agents.agent import Agent
from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.game.const import color_masks, card_strings, card_values
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber


class MyAgent(Agent):
    """
    Sample implementation of a player to play Jass.
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
        #Schaut von welcher Farbe er am meisten Karten hat und wählt diese als trumpf
        for c in range(4):
            #number_in_color zählt wie viele Karten von einer Farbe vorhanden sind
            #color_mask[c] enthält mske für eine gewisse Farbe
            number_in_color = (obs.hand * color_masks[c]).sum()
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
class MyAgent2(Agent):
    """
    Sample implementation of a player to play Jass.
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
        max_number_in_high=0
        max_number_in_low=0
        Ass_Mask=np.array(
            [#  DA DK DQ DJ D10 D9 D8 D7 D6 HA HK HQ HJ H10 H9 H8 H7 H6 SA SK SQ SJ S10 S9 S8 S7 S6 CA CK CQ CJ C10 C9 C8 C7 C6
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,  1, 1, 0, 0, 0, 0, 0, 0, 0,  1, 1, 0, 0, 0, 0, 0, 0,  0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 1, 1,  0, 0, 0, 0, 0, 0, 0, 1,  1]
            ], np.int32)
        if((obs.hand *Ass_Mask[0]).sum()>3):
            return 4
        if((obs.hand *Ass_Mask[1]).sum()>3):
            return 5
        #Schaut von welcher Farbe er am meisten Karten hat und wählt diese als trumpf
        for c in range(4):
            #number_in_color zählt wie viele Karten von einer Farbe vorhanden sind
            #color_mask[c] enthält mske für eine gewisse Farbe
            number_in_color = (obs.hand * color_masks[c]).sum()
            if number_in_color > max_number_in_color:
                max_number_in_color = number_in_color
                trump = c
        return trump

    def action_play_card(self, obs: GameObservation) -> int:
        Ass_Mask_value = np.array(
            [
                # DA DK DQ DJ D10 D9 D8 D7 D6 HA HK HQ HJ H10 H9 H8 H7 H6 SA SK SQ SJ S10 S9 S8 S7 S6 CA CK CQ CJ C10 C9 C8 C7 C6
                [10, 9, 8, 7, 6, 5, 4, 3, 2,  10, 9, 8, 7, 6, 5, 4, 3, 2,  10, 9, 8, 7, 6, 5, 4, 3, 2, 10, 9, 8, 7, 6, 5, 4, 3, 2],
                [ 1, 2, 3, 4, 5,  6, 7, 8, 9,  1, 2, 3, 4, 5,  6, 7, 8, 9,  1, 2, 3, 4, 5,  6, 7, 8, 9, 1, 2, 3, 4, 5,  6, 7, 8, 9]
            ], np.int32)

        # cards are one hot encoded
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        #Wenn ich der erste bin der Spielen darf, spiele ich meine beste karte
        if(obs.nr_cards_in_trick==0):
            if (obs.trump==4):
                myarray = np.array(valid_cards * Ass_Mask_value[0])
                mylist = myarray.tolist()
                card = mylist.index(max(mylist))
                if(max(myarray)>0):
                    return card
            if (obs.trump==5):
                myarray = np.array(valid_cards * Ass_Mask_value[1])
                mylist = myarray.tolist()
                card = mylist.index(max(mylist))
                if(max(myarray)>0):
                    return card
        else:
            #zeigt an welche karten bis jetzt gespielt wurden
            abc=obs.current_trick
            #print(abc)

        # convert to list and draw a value
        card = self._rng.choice(np.flatnonzero(valid_cards))
        self._logger.debug('Played card: {}'.format(card_strings[card]))
        return card

def main():
    # Set the global logging levl (Set to debug or info to see more messages)
    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(nr_games_to_play=10000, save_filename='arena_games')
    player = MyAgent()#AgentRandomSchieber()
    my_player = MyAgent2()

    arena.set_players(my_player, player, my_player, player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0(myAgend2): {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1(myAgend): {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
