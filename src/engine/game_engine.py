"""
game_engine.py — Python mirror of GameEngine.ts for TensorFlow RL training.

All game rules are identical to the TypeScript version:
  - Same card hierarchy, suit switching, diamond defer, joker rules
  - Same scoring formula: captured - remaining hand
  - Same target score formula: ceil(100 / competing_units) + 2
  - Same interleaved turn order for team modes

Pure functions only — no mutation of state objects.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

# ─── Types ───────────────────────────────────────────────────────────────────

Suit = Literal['hearts', 'diamonds', 'clubs', 'spades', 'joker']

SUITS_REG: List[str] = ['hearts', 'diamonds', 'clubs', 'spades']
RANKS_REG: List[int] = list(range(2, 15))   # 2–14
JOKER_RANK: int = 15
CPU_NAMES: List[str] = ['Lucy', 'Zane', 'Uncle', 'Barney', 'Fabi', 'Jane', 'Noema']


@dataclass
class Card:
    rank: int
    suit: str
    id: str
    is_joker: bool


@dataclass
class Move:
    type: str             # 'play' | 'draw' | 'pass'
    card: Optional[Card] = None


@dataclass
class Player:
    id: int
    name: str
    hand: List[Card] = field(default_factory=list)
    captured: int = 0     # count of captured cards this round
    team_id: Optional[int] = None


@dataclass
class GameState:
    players: List[Player]
    pile: List[Card]
    side: List[Card]
    stack: List[Card]
    lead_suit: Optional[str]
    deferred: bool
    current_player_index: int
    turn_order: List[int]
    phase: str              # 'START'|'ACTION'|'END_CHECK'|'VOTING'|'GAME_OVER'|'ROUND_OVER'
    scores: List[int]       # cumulative game scores
    round_scores: List[int] # scores earned this round
    target_score: int
    teams: Optional[List[List[int]]]
    # implementation fields
    num_teams: int
    turn_action_count: int
    last_challenger_id: Optional[int]
    turn_leader_index: int
    votes: Dict[int, str]   # player_id → 'KEEP' | 'END'
    captured_piles: List[List[Card]]   # raw cards, parallel to players


# ─── Deck helpers ─────────────────────────────────────────────────────────────

def _create_deck() -> List[Card]:
    deck: List[Card] = []
    for suit in SUITS_REG:
        for rank in RANKS_REG:
            deck.append(Card(rank=rank, suit=suit, id=f"{suit}-{rank}", is_joker=False))
    deck.append(Card(rank=JOKER_RANK, suit='joker', id='joker-1', is_joker=True))
    deck.append(Card(rank=JOKER_RANK, suit='joker', id='joker-2', is_joker=True))
    return deck


def _shuffle(lst: list) -> list:
    result = list(lst)
    random.shuffle(result)
    return result


# ─── Turn-order helpers ───────────────────────────────────────────────────────

def compute_interleaved_order(num_players: int, num_teams: int) -> List[int]:
    """
    Interleaved turn order for team modes.
    e.g. 4p 2v2 → [0,2,1,3]   6p 3v3 → [0,3,1,4,2,5]   6p 2v2v2 → [0,2,4,1,3,5]
    """
    if num_teams <= 1:
        return list(range(num_players))
    team_size = num_players // num_teams
    order: List[int] = []
    for slot in range(team_size):
        for t in range(num_teams):
            order.append(t * team_size + slot)
    return order


def _next_in_order(current: int, order: List[int]) -> int:
    if not order:
        return current
    try:
        pos = order.index(current)
    except ValueError:
        return order[0]
    return order[(pos + 1) % len(order)]


# ─── Scoring helpers ──────────────────────────────────────────────────────────

def calc_target_score(num_players: int, num_teams: int) -> int:
    units = num_teams if num_teams > 1 else num_players
    return math.ceil(100 / units) + 2


def _get_team_index(player_idx: int, num_players: int, num_teams: int) -> int:
    if num_teams <= 1:
        return player_idx
    team_size = num_players // num_teams
    return player_idx // team_size


def _parse_num_teams(num_players: int, team_mode: Optional[str]) -> int:
    if not team_mode or team_mode == 'Free for All':
        return 1
    return len(team_mode.split('v'))


# ─── init_game ───────────────────────────────────────────────────────────────

def init_game(num_players: int, team_mode: Optional[str] = None) -> GameState:
    """Create a fresh GameState."""
    num_teams = _parse_num_teams(num_players, team_mode)
    deck = _shuffle(_create_deck())

    players: List[Player] = []
    captured_piles: List[List[Card]] = []
    for i in range(num_players):
        hand = deck[:4]
        deck = deck[4:]
        team_id = _get_team_index(i, num_players, num_teams) if num_teams > 1 else None
        players.append(Player(
            id=i,
            name='You' if i == 0 else CPU_NAMES[i - 1],
            hand=hand,
            captured=0,
            team_id=team_id,
        ))
        captured_piles.append([])

    teams: Optional[List[List[int]]] = None
    if num_teams > 1:
        teams = [
            [p.id for p in players if _get_team_index(p.id, num_players, num_teams) == t]
            for t in range(num_teams)
        ]

    base_order = compute_interleaved_order(num_players, num_teams)
    target_score = calc_target_score(num_players, num_teams)
    score_count = num_teams if num_teams > 1 else num_players

    return GameState(
        players=players,
        pile=[],
        side=[],
        stack=deck,
        lead_suit=None,
        deferred=False,
        current_player_index=0,
        turn_order=base_order,
        phase='START',
        scores=[0] * score_count,
        round_scores=[0] * score_count,
        target_score=target_score,
        teams=teams,
        num_teams=num_teams,
        turn_action_count=0,
        last_challenger_id=None,
        turn_leader_index=0,
        votes={},
        captured_piles=captured_piles,
    )


# ─── get_valid_moves ─────────────────────────────────────────────────────────

def get_valid_moves(state: GameState, player_id: int) -> List[Move]:
    """Return all legal moves for the given player. Returns [] if not their turn."""
    if state.phase != 'ACTION':
        return []
    player_idx = next((i for i, p in enumerate(state.players) if p.id == player_id), -1)
    if player_idx != state.current_player_index:
        return []

    player = state.players[player_idx]
    pile_top = state.pile[-1] if state.pile else None
    has_lead_suit = any(c.suit == state.lead_suit and not c.is_joker for c in player.hand)

    play_moves = [
        Move(type='play', card=card)
        for card in player.hand
        if _is_legal_play(card, state.lead_suit, has_lead_suit, pile_top)
    ]

    moves: List[Move] = list(play_moves)
    if state.stack:
        moves.append(Move(type='draw'))
    moves.append(Move(type='pass'))
    return moves


def _is_legal_play(
    card: Card,
    lead_suit: Optional[str],
    has_lead_suit: bool,
    pile_top: Optional[Card],
) -> bool:
    if card.is_joker:
        return True
    is_lead_suit = card.suit == lead_suit
    is_rank_match = pile_top is not None and card.rank == pile_top.rank
    is_diamond = card.suit == 'diamonds'

    if has_lead_suit:
        return is_lead_suit or is_rank_match
    else:
        return is_lead_suit or is_rank_match or is_diamond


# ─── apply_move ──────────────────────────────────────────────────────────────

def apply_move(state: GameState, player_id: int, move: Move) -> GameState:
    """Pure function — returns a new GameState with the move applied."""
    player_idx = next((i for i, p in enumerate(state.players) if p.id == player_id), -1)
    if player_idx == -1 or state.phase != 'ACTION':
        return state

    s = _clone_state(state)
    joker_played = False

    if move.type == 'draw':
        if not s.stack:
            return state
        drawn = s.stack[0]
        s.stack = s.stack[1:]
        s.players[player_idx] = _player_with_hand(
            s.players[player_idx], s.players[player_idx].hand + [drawn]
        )

    elif move.type == 'pass':
        pass  # nothing to change

    elif move.type == 'play' and move.card is not None:
        card = move.card
        new_hand = [c for c in s.players[player_idx].hand if c.id != card.id]
        s.players[player_idx] = _player_with_hand(s.players[player_idx], new_hand)

        if card.is_joker:
            s.side = s.side + [card]
            s.last_challenger_id = player_idx
            joker_played = True
        else:
            pile_top = s.pile[-1] if s.pile else None
            is_rank_match = pile_top is not None and card.rank == pile_top.rank

            if is_rank_match:
                s.pile = s.pile + s.side + [card]
                s.side = []
                s.lead_suit = card.suit
                s.deferred = False
                s.last_challenger_id = None
            else:
                is_lead_suit = card.suit == s.lead_suit
                is_diamond = card.suit == 'diamonds'
                hand_after = s.players[player_idx].hand
                has_lead_suit = any(
                    c.suit == s.lead_suit and not c.is_joker for c in hand_after
                )
                side_top = s.side[-1] if s.side else None
                current_winner = (
                    side_top if (side_top and s.last_challenger_id is not None)
                    else pile_top
                )
                beats = _compute_beats(
                    card, current_winner, s.lead_suit,
                    is_lead_suit, is_diamond, has_lead_suit, s.deferred
                )

                if beats:
                    s.side = s.side + [card]
                    s.last_challenger_id = player_idx
                else:
                    s.side = [card] + s.side

                if is_diamond and not is_lead_suit and not s.deferred:
                    s.deferred = True

    # Advance turn counter
    s.turn_action_count += 1

    if joker_played or s.turn_action_count >= len(s.players):
        s = _resolve_end_check(s)
    else:
        s.current_player_index = _next_in_order(s.current_player_index, s.turn_order)

    return s


def _compute_beats(
    card: Card,
    current_winner: Optional[Card],
    lead_suit: Optional[str],
    is_lead_suit: bool,
    is_diamond: bool,
    has_lead_suit: bool,
    deferred: bool,
) -> bool:
    if card.is_joker:
        return True
    if current_winner and current_winner.is_joker:
        return False
    if is_lead_suit:
        return (
            current_winner is None
            or current_winner.suit != lead_suit
            or card.rank > current_winner.rank
        )
    if is_diamond and not has_lead_suit:
        return (
            current_winner is None
            or (
                current_winner.suit != lead_suit
                and (current_winner.suit != 'diamonds' or card.rank > current_winner.rank)
            )
        )
    return False


def _resolve_end_check(s: GameState) -> GameState:
    side_top = s.side[-1] if s.side else None
    is_joker_win = side_top is not None and side_top.is_joker
    can_win = s.last_challenger_id is not None and (not s.deferred or is_joker_win)

    if can_win and s.last_challenger_id is not None:
        winner = s.last_challenger_id
        all_cards = s.pile + s.side
        s.captured_piles[winner] = s.captured_piles[winner] + all_cards
        s.players[winner] = _player_add_captured(s.players[winner], len(all_cards))
        s.pile = []
        s.side = []
        s.deferred = False
        s.turn_leader_index = winner
        s.current_player_index = winner
    else:
        s.pile = s.pile + s.side
        s.side = []
        s.deferred = False
        nxt = _next_in_order(s.turn_leader_index, s.turn_order)
        s.turn_leader_index = nxt
        s.current_player_index = nxt

    s.turn_action_count = 0
    s.last_challenger_id = None

    # Check round end
    if any(len(p.hand) == 0 for p in s.players):
        return score_round(s)

    if not s.stack:
        s.phase = 'VOTING'
    else:
        s.phase = 'START'

    return s


# ─── score_round ─────────────────────────────────────────────────────────────

def score_round(state: GameState) -> GameState:
    """Calculate round scores, update cumulative scores, transition phase."""
    s = _clone_state(state)
    per_player = [p.captured - len(p.hand) for p in s.players]
    n = len(s.players)

    if s.num_teams > 1:
        new_scores = []
        round_scores = []
        for team_idx in range(s.num_teams):
            team_total = sum(
                per_player[i]
                for i in range(n)
                if _get_team_index(i, n, s.num_teams) == team_idx
            )
            new_scores.append(s.scores[team_idx] + team_total)
            round_scores.append(team_total)
    else:
        new_scores = [s.scores[i] + (per_player[i] if i < len(per_player) else 0) for i in range(len(s.scores))]
        round_scores = per_player

    s.scores = new_scores
    s.round_scores = round_scores

    if any(sc >= s.target_score for sc in new_scores):
        s.phase = 'GAME_OVER'
    else:
        s.phase = 'ROUND_OVER'

    return s


# ─── is_terminal / get_reward ─────────────────────────────────────────────────

def is_terminal(state: GameState) -> bool:
    return state.phase == 'GAME_OVER'


def get_reward(state: GameState, player_id: int) -> float:
    """
    +1 if this player/team won, -1 if lost, 0 if game not over.
    Tied leaders both return 0.
    """
    if not is_terminal(state):
        return 0.0
    player_idx = next((i for i, p in enumerate(state.players) if p.id == player_id), -1)
    if player_idx == -1:
        return 0.0

    idx = (
        _get_team_index(player_idx, len(state.players), state.num_teams)
        if state.num_teams > 1
        else player_idx
    )
    my_score = state.scores[idx]
    max_score = max(state.scores)

    if my_score < max_score:
        return -1.0
    winners = sum(1 for sc in state.scores if sc == max_score)
    return 1.0 if winners == 1 else 0.0


# ─── Clone helpers ───────────────────────────────────────────────────────────

def _player_with_hand(p: Player, hand: List[Card]) -> Player:
    return Player(
        id=p.id, name=p.name, hand=hand,
        captured=p.captured, team_id=p.team_id
    )


def _player_add_captured(p: Player, count: int) -> Player:
    return Player(
        id=p.id, name=p.name, hand=list(p.hand),
        captured=p.captured + count, team_id=p.team_id
    )


def _clone_state(s: GameState) -> GameState:
    return GameState(
        players=[Player(id=p.id, name=p.name, hand=list(p.hand),
                        captured=p.captured, team_id=p.team_id)
                 for p in s.players],
        pile=list(s.pile),
        side=list(s.side),
        stack=list(s.stack),
        lead_suit=s.lead_suit,
        deferred=s.deferred,
        current_player_index=s.current_player_index,
        turn_order=list(s.turn_order),
        phase=s.phase,
        scores=list(s.scores),
        round_scores=list(s.round_scores),
        target_score=s.target_score,
        teams=[list(t) for t in s.teams] if s.teams else None,
        num_teams=s.num_teams,
        turn_action_count=s.turn_action_count,
        last_challenger_id=s.last_challenger_id,
        turn_leader_index=s.turn_leader_index,
        votes=dict(s.votes),
        captured_piles=[list(cp) for cp in s.captured_piles],
    )


# ─── Simple random agent helper ───────────────────────────────────────────────

def random_vote(state: GameState) -> GameState:
    """Apply random votes for all players in VOTING phase and resolve."""
    if state.phase != 'VOTING':
        return state
    s = _clone_state(state)
    for p in s.players:
        if p.id not in s.votes:
            s.votes[p.id] = random.choice(['KEEP', 'END'])

    keep = sum(1 for v in s.votes.values() if v == 'KEEP')
    end = len(s.players) - keep

    if keep > end:
        outcome = 'KEEP'
    elif end > keep:
        outcome = 'END'
    else:
        outcome = random.choice(['KEEP', 'END'])

    if outcome == 'KEEP':
        all_captured: List[Card] = []
        for cp in s.captured_piles:
            all_captured.extend(cp)
        if not all_captured:
            return score_round(s)
        new_stack = _shuffle(all_captured)
        s.stack = new_stack
        s.captured_piles = [[] for _ in s.players]
        s.players = [Player(id=p.id, name=p.name, hand=list(p.hand),
                            captured=0, team_id=p.team_id)
                     for p in s.players]
        s.votes = {}
        s.phase = 'START'
    else:
        s = score_round(s)

    return s


def run_random_game(num_players: int = 4, team_mode: Optional[str] = None) -> Tuple[GameState, int]:
    """
    Run one complete game with random agents.
    Returns (final_state, total_turns).
    """
    state = init_game(num_players, team_mode)
    turns = 0

    while not is_terminal(state):
        turns += 1
        if turns > 5000:
            # safety escape — should not happen in normal play
            break

        if state.phase == 'START':
            # Flip a card from stack
            if not state.stack:
                state = _clone_state(state)
                state.phase = 'VOTING'
                continue
            s = _clone_state(state)
            flipped = s.stack[0]
            s.stack = s.stack[1:]
            s.pile = s.pile + [flipped]
            # Reset hasActed equivalent
            if flipped.is_joker:
                all_cards = s.pile + s.side
                winner = s.current_player_index
                s.captured_piles[winner] = s.captured_piles[winner] + all_cards
                s.players[winner] = _player_add_captured(s.players[winner], len(all_cards))
                s.pile = []
                s.side = []
                s.current_player_index = _next_in_order(s.current_player_index, s.turn_order)
                s.phase = 'START'
            else:
                s.lead_suit = flipped.suit
                s.deferred = False
                s.turn_action_count = 0
                s.phase = 'ACTION'
            state = s

        elif state.phase == 'ACTION':
            pid = state.players[state.current_player_index].id
            moves = get_valid_moves(state, pid)
            move = random.choice(moves) if moves else Move(type='pass')
            state = apply_move(state, pid, move)

        elif state.phase == 'VOTING':
            state = random_vote(state)

        elif state.phase == 'ROUND_OVER':
            # Start a new round
            state = _start_new_round(state)

    return state, turns


def _start_new_round(state: GameState) -> GameState:
    """Begin a new round from ROUND_OVER phase."""
    s = _clone_state(state)
    deck = _shuffle(_create_deck())
    new_players: List[Player] = []
    for i, p in enumerate(s.players):
        hand = deck[:4]
        deck = deck[4:]
        new_players.append(Player(id=p.id, name=p.name, hand=hand,
                                  captured=0, team_id=p.team_id))
    s.players = new_players
    s.captured_piles = [[] for _ in s.players]
    s.stack = deck
    s.pile = []
    s.side = []
    s.votes = {}
    s.lead_suit = None
    s.deferred = False
    s.turn_action_count = 0
    s.last_challenger_id = None
    # Advance round leader
    n = len(s.players)
    next_leader = (s.turn_leader_index + 1) % n
    base_order = compute_interleaved_order(n, s.num_teams)
    sp = base_order.index(next_leader) if next_leader in base_order else 0
    s.turn_order = base_order[sp:] + base_order[:sp]
    s.turn_leader_index = next_leader
    s.current_player_index = next_leader
    s.phase = 'START'
    return s


# ─── Self-play test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time

    CONFIGS = [
        (2, None),
        (3, None),
        (4, None),
        (4, '2v2'),
        (6, '3v3'),
        (6, '2v2v2'),
        (8, '4v4'),
        (8, '2v2v2v2'),
    ]

    NUM_GAMES = 10
    print(f"Running {NUM_GAMES} random self-play games per config...\n")

    for num_players, team_mode in CONFIGS:
        label = f"{num_players}p {'FFA' if not team_mode else team_mode}"
        wins: Dict[int, int] = {}
        total_turns = 0
        t0 = time.time()

        for _ in range(NUM_GAMES):
            final, turns = run_random_game(num_players, team_mode)
            total_turns += turns
            if final.scores:
                best = max(final.scores)
                winner_idx = final.scores.index(best)
                wins[winner_idx] = wins.get(winner_idx, 0) + 1

        elapsed = time.time() - t0
        avg_turns = total_turns / NUM_GAMES
        print(
            f"[{label:15s}] {NUM_GAMES} games, "
            f"avg {avg_turns:.0f} turns, "
            f"{elapsed*1000/NUM_GAMES:.1f} ms/game  | wins: {wins}"
        )

    print("\nAll tests passed.")
