/**
 * GameEngine.ts — Pure TypeScript game engine for Deference.
 * Zero React dependencies. Suitable for RL training and simulation.
 *
 * Phase 1 of RL Training extraction from App.tsx.
 */

// ─── Types ──────────────────────────────────────────────────────────────────

export type Suit = 'hearts' | 'diamonds' | 'clubs' | 'spades' | 'joker'

export type Card = {
  rank: number   // 2–14 regular, 15 = joker
  suit: Suit
  id: string
  isJoker: boolean
}

export type Move =
  | { type: 'play'; card: Card }
  | { type: 'draw' }
  | { type: 'pass' }

export type Player = {
  id: number
  name: string
  hand: Card[]
  captured: number   // count of captured cards this round
  teamId?: number
}

export type GameState = {
  players: Player[]
  pile: Card[]
  side: Card[]
  stack: Card[]
  leadSuit: Suit | null
  deferred: boolean
  currentPlayerIndex: number
  turnOrder: number[]
  phase: 'START' | 'ACTION' | 'END_CHECK' | 'VOTING' | 'GAME_OVER' | 'ROUND_OVER'
  scores: number[]        // cumulative game scores (per team or per player)
  roundScores: number[]   // scores earned in the current round
  targetScore: number
  teams?: number[][]      // teams[teamIdx] = [playerIdx, ...]
  // ── implementation fields ──
  numTeams: number
  turnActionCount: number
  lastChallengerId: number | null
  turnLeaderIndex: number
  votes: Record<number, 'KEEP' | 'END'>
  /** Raw captured cards — parallel to players array. Needed for reshuffling. */
  capturedPiles: Card[][]
}

// ─── Constants ───────────────────────────────────────────────────────────────

const SUITS_REG: Suit[] = ['hearts', 'diamonds', 'clubs', 'spades']
const RANKS_REG = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
const JOKER_RANK = 15
const CPU_NAMES = ['Lucy', 'Zane', 'Uncle', 'Barney', 'Fabi', 'Jane', 'Noema']

// ─── Deck helpers ─────────────────────────────────────────────────────────────

function createDeck(): Card[] {
  const deck: Card[] = []
  for (const suit of SUITS_REG) {
    for (const rank of RANKS_REG) {
      deck.push({ id: `${suit}-${rank}`, suit, rank, isJoker: false })
    }
  }
  deck.push({ id: 'joker-1', suit: 'joker', rank: JOKER_RANK, isJoker: true })
  deck.push({ id: 'joker-2', suit: 'joker', rank: JOKER_RANK, isJoker: true })
  return deck
}

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

// ─── Turn-order helpers ───────────────────────────────────────────────────────

/**
 * Compute interleaved turn order for team modes.
 * Teams are sequential blocks of size (numPlayers / numTeams).
 * e.g. 4p 2v2 → [0,2,1,3]   6p 3v3 → [0,3,1,4,2,5]   6p 2v2v2 → [0,2,4,1,3,5]
 */
export function computeInterleavedOrder(numPlayers: number, numTeams: number): number[] {
  if (numTeams <= 1) return Array.from({ length: numPlayers }, (_, i) => i)
  const teamSize = Math.floor(numPlayers / numTeams)
  const order: number[] = []
  for (let slot = 0; slot < teamSize; slot++) {
    for (let t = 0; t < numTeams; t++) {
      order.push(t * teamSize + slot)
    }
  }
  return order
}

function nextInOrder(current: number, order: number[]): number {
  if (order.length === 0) return current
  const pos = order.indexOf(current)
  if (pos === -1) return order[0]
  return order[(pos + 1) % order.length]
}

// ─── Scoring helpers ──────────────────────────────────────────────────────────

export function calcTargetScore(numPlayers: number, numTeams: number): number {
  const units = numTeams > 1 ? numTeams : numPlayers
  return Math.ceil(100 / units) + 2
}

function getTeamIndex(playerIdx: number, numPlayers: number, numTeams: number): number {
  if (numTeams <= 1) return playerIdx
  const teamSize = Math.floor(numPlayers / numTeams)
  return Math.floor(playerIdx / teamSize)
}

// ─── initGame ────────────────────────────────────────────────────────────────

/**
 * Create a fresh GameState for the given player count and optional team mode string.
 * teamMode: '2v2' | '3v3' | '2v2v2' | '4v4' | '2v2v2v2' | null (FFA)
 */
export function initGame(numPlayers: number, teamMode?: string): GameState {
  const numTeams = parseNumTeams(numPlayers, teamMode)
  const deck = shuffle(createDeck())

  const players: Player[] = []
  const capturedPiles: Card[][] = []
  for (let i = 0; i < numPlayers; i++) {
    players.push({
      id: i,
      name: i === 0 ? 'You' : CPU_NAMES[i - 1],
      hand: deck.splice(0, 4),
      captured: 0,
      teamId: numTeams > 1 ? getTeamIndex(i, numPlayers, numTeams) : undefined,
    })
    capturedPiles.push([])
  }

  const teams: number[][] | undefined = numTeams > 1
    ? Array.from({ length: numTeams }, (_, t) =>
        players.filter((_, i) => getTeamIndex(i, numPlayers, numTeams) === t).map(p => p.id))
    : undefined

  const baseOrder = computeInterleavedOrder(numPlayers, numTeams)
  const targetScore = calcTargetScore(numPlayers, numTeams)
  const scoreCount = numTeams > 1 ? numTeams : numPlayers

  return {
    players,
    pile: [],
    side: [],
    stack: deck,
    leadSuit: null,
    deferred: false,
    currentPlayerIndex: 0,
    turnOrder: baseOrder,
    phase: 'START',
    scores: new Array(scoreCount).fill(0),
    roundScores: new Array(scoreCount).fill(0),
    targetScore,
    teams,
    numTeams,
    turnActionCount: 0,
    lastChallengerId: null,
    turnLeaderIndex: 0,
    votes: {},
    capturedPiles,
  }
}

function parseNumTeams(numPlayers: number, teamMode?: string): number {
  if (!teamMode || teamMode === 'Free for All') return 1
  const parts = teamMode.split('v')
  return parts.length
}

// ─── getValidMoves ───────────────────────────────────────────────────────────

/**
 * Return all legal moves for the given player in the current state.
 * Returns [] if it's not this player's turn or the phase isn't ACTION.
 */
export function getValidMoves(state: GameState, playerId: number): Move[] {
  if (state.phase !== 'ACTION') return []
  const playerIdx = state.players.findIndex(p => p.id === playerId)
  if (playerIdx !== state.currentPlayerIndex) return []

  const player = state.players[playerIdx]
  const pileTop = state.pile.length > 0 ? state.pile[state.pile.length - 1] : null
  const hasLeadSuit = player.hand.some(c => c.suit === state.leadSuit && !c.isJoker)

  const playMoves: Move[] = player.hand
    .filter(card => isLegalPlay(card, state.leadSuit, hasLeadSuit, pileTop))
    .map(card => ({ type: 'play', card }))

  const moves: Move[] = [...playMoves]
  if (state.stack.length > 0) moves.push({ type: 'draw' })
  moves.push({ type: 'pass' })
  return moves
}

function isLegalPlay(
  card: Card,
  leadSuit: Suit | null,
  hasLeadSuit: boolean,
  pileTop: Card | null,
): boolean {
  if (card.isJoker) return true
  const isLeadSuit = card.suit === leadSuit
  const isRankMatch = pileTop !== null && card.rank === pileTop.rank
  const isDiamond = card.suit === 'diamonds'

  if (hasLeadSuit) {
    return isLeadSuit || isRankMatch
  } else {
    return isLeadSuit || isRankMatch || isDiamond || card.isJoker
  }
}

// ─── applyMove ───────────────────────────────────────────────────────────────

/**
 * Pure function — applies a move and returns a new GameState.
 * Handles full phase transitions including END_CHECK resolution.
 */
export function applyMove(state: GameState, playerId: number, move: Move): GameState {
  const playerIdx = state.players.findIndex(p => p.id === playerId)
  if (playerIdx === -1 || state.phase !== 'ACTION') return state

  let s = cloneState(state)
  let jokerPlayed = false

  if (move.type === 'draw') {
    if (s.stack.length === 0) return state
    const drawn = s.stack[0]
    s.stack = s.stack.slice(1)
    s.players = s.players.map((p, i) =>
      i === playerIdx ? { ...p, hand: [...p.hand, drawn] } : p
    )
  } else if (move.type === 'pass') {
    // nothing to change for the player
  } else if (move.type === 'play') {
    const card = move.card
    // Remove from hand
    s.players = s.players.map((p, i) =>
      i === playerIdx ? { ...p, hand: p.hand.filter(c => c.id !== card.id) } : p
    )

    if (card.isJoker) {
      s.side = [...s.side, card]
      s.lastChallengerId = playerIdx
      jokerPlayed = true
    } else {
      const pileTop = s.pile.length > 0 ? s.pile[s.pile.length - 1] : null
      const isRankMatch = pileTop !== null && card.rank === pileTop.rank

      if (isRankMatch) {
        // Suit switch: merge side into pile
        s.pile = [...s.pile, ...s.side, card]
        s.side = []
        s.leadSuit = card.suit
        s.deferred = false
        s.lastChallengerId = null
      } else {
        const isLeadSuit = card.suit === s.leadSuit
        const isDiamond = card.suit === 'diamonds'
        // Remaining hand after play (for hasLeadSuit check)
        const handAfter = s.players[playerIdx].hand
        const hasLeadSuit = handAfter.some(c => c.suit === s.leadSuit && !c.isJoker)

        const sideTop = s.side.length > 0 ? s.side[s.side.length - 1] : null
        const currentWinner = (sideTop && s.lastChallengerId !== null) ? sideTop : pileTop

        const beats = computeBeats(card, currentWinner, s.leadSuit, isLeadSuit, isDiamond, hasLeadSuit, s.deferred)

        if (beats) {
          s.side = [...s.side, card]
          s.lastChallengerId = playerIdx
        } else {
          s.side = [card, ...s.side]
        }

        if (isDiamond && !isLeadSuit && !s.deferred) {
          s.deferred = true
        }
      }
    }
  }

  // Advance turn counter
  const newCount = s.turnActionCount + 1
  s.turnActionCount = newCount

  if (jokerPlayed || newCount >= s.players.length) {
    s = resolveEndCheck(s)
  } else {
    s.currentPlayerIndex = nextInOrder(s.currentPlayerIndex, s.turnOrder)
  }

  return s
}

function computeBeats(
  card: Card,
  currentWinner: Card | null,
  leadSuit: Suit | null,
  isLeadSuit: boolean,
  isDiamond: boolean,
  hasLeadSuit: boolean,
  deferred: boolean,
): boolean {
  if (card.isJoker) return true
  if (currentWinner?.isJoker) return false
  if (isLeadSuit) {
    return !currentWinner || currentWinner.suit !== leadSuit || card.rank > currentWinner.rank
  }
  if (isDiamond && !hasLeadSuit) {
    return !currentWinner || (
      currentWinner.suit !== leadSuit &&
      (currentWinner.suit !== 'diamonds' || card.rank > currentWinner.rank)
    )
  }
  return false
}

function resolveEndCheck(s: GameState): GameState {
  const sideTop = s.side.length > 0 ? s.side[s.side.length - 1] : null
  const isJokerWin = sideTop?.isJoker === true
  const canWin = s.lastChallengerId !== null && (!s.deferred || isJokerWin)

  if (canWin && s.lastChallengerId !== null) {
    const allCards = [...s.pile, ...s.side]
    const winner = s.lastChallengerId
    const newPiles = s.capturedPiles.map((pile, i) =>
      i === winner ? [...pile, ...allCards] : pile
    )
    s.players = s.players.map((p, i) =>
      i === winner ? { ...p, captured: p.captured + allCards.length } : p
    )
    s.capturedPiles = newPiles
    s.pile = []
    s.side = []
    s.deferred = false
    s.turnLeaderIndex = winner
    s.currentPlayerIndex = winner
  } else {
    s.pile = [...s.pile, ...s.side]
    s.side = []
    s.deferred = false
    const next = nextInOrder(s.turnLeaderIndex, s.turnOrder)
    s.turnLeaderIndex = next
    s.currentPlayerIndex = next
  }

  s.turnActionCount = 0
  s.lastChallengerId = null

  // Check round end
  const someoneOut = s.players.some(p => p.hand.length === 0)
  if (someoneOut) {
    return scoreRound(s)
  }

  if (s.stack.length === 0) {
    s.phase = 'VOTING'
  } else {
    s.phase = 'START'
  }
  return s
}

// ─── scoreRound ──────────────────────────────────────────────────────────────

/**
 * Calculate round scores, add to cumulative scores, and transition phase.
 * Returns new state with phase = 'GAME_OVER' or 'ROUND_OVER'.
 */
export function scoreRound(state: GameState): GameState {
  const s = cloneState(state)
  const { players, numTeams, scores, targetScore } = s
  const perPlayer = players.map(p => p.captured - p.hand.length)

  let newScores: number[]
  let roundScores: number[]

  if (numTeams > 1) {
    newScores = scores.map((base, teamIdx) => {
      const teamTotal = perPlayer
        .filter((_, i) => getTeamIndex(i, players.length, numTeams) === teamIdx)
        .reduce((a, b) => a + b, 0)
      return base + teamTotal
    })
    roundScores = newScores.map((s, i) => s - scores[i])
  } else {
    newScores = scores.map((base, i) => base + (perPlayer[i] ?? 0))
    roundScores = perPlayer
  }

  s.scores = newScores
  s.roundScores = roundScores

  if (newScores.some(sc => sc >= targetScore)) {
    s.phase = 'GAME_OVER'
  } else {
    s.phase = 'ROUND_OVER'
  }

  return s
}

// ─── isTerminal ──────────────────────────────────────────────────────────────

/** Returns true if the game has ended. */
export function isTerminal(state: GameState): boolean {
  return state.phase === 'GAME_OVER'
}

// ─── getReward ───────────────────────────────────────────────────────────────

/**
 * +1 if this player/team won, -1 if lost, 0 if game is not over.
 * In multi-way contests, winning = having the highest score.
 */
export function getReward(state: GameState, playerId: number): number {
  if (!isTerminal(state)) return 0

  const playerIdx = state.players.findIndex(p => p.id === playerId)
  if (playerIdx === -1) return 0

  const { scores, numTeams } = state
  const idx = numTeams > 1
    ? getTeamIndex(playerIdx, state.players.length, numTeams)
    : playerIdx

  const myScore = scores[idx]
  const maxScore = Math.max(...scores)

  if (myScore < maxScore) return -1
  // Winner (or tied for top)
  const winners = scores.filter(s => s === maxScore).length
  return winners === 1 ? 1 : 0   // ties return 0 (no winner yet in practice — target not shared)
}

// ─── Deep-clone helper ────────────────────────────────────────────────────────

function cloneState(s: GameState): GameState {
  return {
    ...s,
    players: s.players.map(p => ({ ...p, hand: [...p.hand] })),
    pile: [...s.pile],
    side: [...s.side],
    stack: [...s.stack],
    turnOrder: [...s.turnOrder],
    scores: [...s.scores],
    roundScores: [...s.roundScores],
    votes: { ...s.votes },
    capturedPiles: s.capturedPiles.map(cp => [...cp]),
    teams: s.teams ? s.teams.map(t => [...t]) : undefined,
  }
}
