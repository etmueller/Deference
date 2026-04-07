/**
 * aiInference.ts — In-browser ONNX inference for Deference AI players.
 *
 * Runs entirely client-side using onnxruntime-web. No Python server needed.
 *
 * Install:
 *   npm install onnxruntime-web
 *
 * Usage in App.tsx:
 *   import { loadPlayer, getAIMove } from './aiInference'
 *
 *   // At app startup (once per player):
 *   await loadPlayer('Lucy', '/models/Lucy_2800.onnx')
 *
 *   // When it's Lucy's turn (phase === 'ACTION'):
 *   const move = await getAIMove('Lucy', appState, currentPlayerIndex)
 *   handleAction(move.type, move.card)
 */

import * as ort from 'onnxruntime-web'

// Tell ort where to find its WASM files (must match your base path)
//ort.env.wasm.wasmPaths = {
//  wasm: '/Deference/ort-wasm-simd-threaded.wasm',
//  mjs:  '/Deference/ort-wasm-simd-threaded.mjs',
//}
ort.env.wasm.numThreads = 1

ort.env.wasm.numThreads = 1
ort.env.wasm.simd = false

// ─── Types (mirrors App.tsx — keep in sync) ──────────────────────────────────

type Suit = 'H' | 'D' | 'C' | 'S' | 'J'

type Card = {
  id: string       // e.g. "H-11", "D-2", "JOKER-1"
  suit: Suit
  rank: number     // 2–14 regular, 15 = joker
  isJoker: boolean
}

type Player = {
  id: number
  name: string
  hand: Card[]
  captured: Card[]
  isAI: boolean
  hasActed: boolean
}

/** Subset of App.tsx state fields needed for inference. */
export type InferenceState = {
  players: Player[]
  pile: Card[]
  side: Card[]
  deck: Card[]              // the stack
  leadSuit: Suit | null
  deferred: boolean
  currentPlayerIndex: number
  turnOrder: number[]
  turnLeaderIndex: number
  lastChallengerId: number | null
  turnActionCount: number
  gameScores: number[]
  targetScore: number
  numTeams: number
  teams?: number[][]        // teams[teamIdx] = [playerIdx, ...]
  phase: string
  votes: Record<number, 'KEEP' | 'END'>
}

export type AIMove =
  | { type: 'PLAY'; card: Card }
  | { type: 'DRAW' }
  | { type: 'PASS' }
  | { type: 'VOTE'; choice: 'KEEP' | 'END' }


// ─── Constants ───────────────────────────────────────────────────────────────

const NUM_CARDS   = 54
const ACTION_SIZE = 58
const ACTION_DRAW          = 54
const ACTION_PASS          = 55
const ACTION_VOTE_CONTINUE = 56
const ACTION_VOTE_END      = 57
const MAX_PLAYERS = 8
const STATE_DIM   = 392   // base encode_observation output
const STATE_DIM_V2 = 572  // encode_observation_v2 output
// obs_input_dim_v2 = STATE_DIM_V2 + ACTION_SIZE = 630

// ─── Player style IDs ────────────────────────────────────────────────────────
// Must match STYLE_TO_ID in training code.
// style_id is embedded per-player — each .onnx file was exported with style baked in
// via the style_ids input. We pass the player's id here.
const STYLE_IDS: Record<string, number> = {
  Lucy:  1,
  Zane:  0,
  Uncle: 4,
  Barney:5,
  Fabi:  6,
  Jane:  3,
  Noema: 2,
}


// ─── Card ID mapping tables ───────────────────────────────────────────────────
// diamonds_env_v2 deck order: clubs 2–A, diamonds 2–A, hearts 2–A, spades 2–A, joker×2
// App.tsx suit codes: C, D, H, S, J

const UI_ID_TO_ENV_INT: Record<string, number> = {
  "C-2":0,"C-3":1,"C-4":2,"C-5":3,"C-6":4,"C-7":5,"C-8":6,"C-9":7,
  "C-10":8,"C-11":9,"C-12":10,"C-13":11,"C-14":12,
  "D-2":13,"D-3":14,"D-4":15,"D-5":16,"D-6":17,"D-7":18,"D-8":19,"D-9":20,
  "D-10":21,"D-11":22,"D-12":23,"D-13":24,"D-14":25,
  "H-2":26,"H-3":27,"H-4":28,"H-5":29,"H-6":30,"H-7":31,"H-8":32,"H-9":33,
  "H-10":34,"H-11":35,"H-12":36,"H-13":37,"H-14":38,
  "S-2":39,"S-3":40,"S-4":41,"S-5":42,"S-6":43,"S-7":44,"S-8":45,"S-9":46,
  "S-10":47,"S-11":48,"S-12":49,"S-13":50,"S-14":51,
  "JOKER-1":52,"JOKER-2":53,
}

// Reverse: env integer → App.tsx card id
const ENV_INT_TO_UI_ID: string[] = new Array(NUM_CARDS)
for (const [uid, eid] of Object.entries(UI_ID_TO_ENV_INT)) {
  ENV_INT_TO_UI_ID[eid] = uid
}

function cardToEnvInt(card: Card): number {
  const id = UI_ID_TO_ENV_INT[card.id]
  if (id === undefined) throw new Error(`Unknown card id: ${card.id}`)
  return id
}

// ─── Env suit index (for one-hot encoding) ────────────────────────────────────
// diamonds_env_v2 Suit enum order used in lead_suit_features():
// [CLUBS, DIAMONDS, HEARTS, SPADES, JOKER] → indices 0–4
const UI_SUIT_TO_ENV_IDX: Record<string, number> = {
  'C': 0,  // clubs
  'D': 1,  // diamonds
  'H': 2,  // hearts
  'S': 3,  // spades
  'J': 4,  // joker
}

// Phase → env Phase enum index
// Phase order in diamonds_env_v2: ROUND_SETUP=0, TURN_FLIP=1, TURN_ACTIONS=2,
// TURN_RESOLUTION=3, ROUND_SCORING=4, VOTE=5, GAME_OVER=6
const PHASE_TO_ENV_IDX: Record<string, number> = {
  'START':      1,  // TURN_FLIP
  'ACTION':     2,  // TURN_ACTIONS
  'END_CHECK':  3,  // TURN_RESOLUTION
  'ROUND_OVER': 4,  // ROUND_SCORING
  'VOTING':     5,  // VOTE
  'GAME_OVER':  6,  // GAME_OVER
}
const NUM_PHASES = 7


// ─── Session registry ─────────────────────────────────────────────────────────

const SESSIONS: Map<string, ort.InferenceSession> = new Map()

/**
 * Load a player's ONNX model. Call once at startup per player.
 * @param playerName  e.g. "Lucy"
 * @param modelUrl    URL to the .onnx file, e.g. "/models/Lucy_2800.onnx"
 */

export async function loadPlayer(playerName: string, modelUrl: string): Promise<void> {
  if (SESSIONS.has(playerName)) return
  
  // Fetch the model as ArrayBuffer and pass directly — avoids external data path issues
  const response = await fetch(modelUrl)
  const arrayBuffer = await response.arrayBuffer()
  
  const session = await ort.InferenceSession.create(arrayBuffer, {
    executionProviders: ['cpu'],
  })
  SESSIONS.set(playerName, session)
  console.log(`[AI] Loaded ${playerName}`)
}

function getSession(playerName: string): ort.InferenceSession {
  const session = SESSIONS.get(playerName)
  if (!session) throw new Error(`Model for ${playerName} not loaded. Call loadPlayer() first.`)
  return session
}


// ─── Observation encoder ─────────────────────────────────────────────────────
// TypeScript port of diamonds_env_v2.py: encode_observation() + encode_observation_v2()
// Produces a 572-float vector. Mask (58 floats) is appended → 630 total for the model.

function safeDiv(n: number, d: number): number {
  return d === 0 ? 0 : n / d
}

function multiHot(cardInts: number[], size = NUM_CARDS): number[] {
  const v = new Array(size).fill(0)
  for (const c of cardInts) v[c] = 1
  return v
}

function oneHotCard(cardInt: number | null, size = NUM_CARDS): number[] {
  const v = new Array(size).fill(0)
  if (cardInt !== null) v[cardInt] = 1
  return v
}

function oneHotPlayer(pid: number | null): number[] {
  const v = new Array(MAX_PLAYERS).fill(0)
  if (pid !== null && pid >= 0 && pid < MAX_PLAYERS) v[pid] = 1
  return v
}

function phaseOneHot(phaseStr: string): number[] {
  const v = new Array(NUM_PHASES).fill(0)
  const idx = PHASE_TO_ENV_IDX[phaseStr] ?? 2
  v[idx] = 1
  return v
}

function leadSuitFeatures(suit: Suit | null): number[] {
  const v = [0, 0, 0, 0, 0]
  if (suit !== null) {
    const idx = UI_SUIT_TO_ENV_IDX[suit]
    if (idx !== undefined) v[idx] = 1
  }
  return v
}

/** Returns true if cardA (env int) beats cardB (env int) in the game hierarchy. */
function beats(cardA: number, cardB: number, leadSuitIdx: number): boolean {
  const isJokerA = cardA >= 52
  const isJokerB = cardB >= 52
  if (isJokerA) return true
  if (isJokerB) return false
  const suitA = Math.floor(cardA / 13)   // 0=clubs,1=diamonds,2=hearts,3=spades
  const suitB = Math.floor(cardB / 13)
  const rankA = (cardA % 13) + 2         // 2–14
  const rankB = (cardB % 13) + 2
  return suitA === suitB && rankA > rankB
}

/** The env integer of the current winning card (side top vs pile top). */
function currentWinningCardInt(
  pileInts: number[],
  sideInts: number[],
  lastChallengerId: number | null,
): number | null {
  const pileTop = pileInts.length > 0 ? pileInts[pileInts.length - 1] : null
  const sideTop = sideInts.length > 0 ? sideInts[sideInts.length - 1] : null
  if (sideTop === null) return pileTop
  if (pileTop === null) return sideTop
  // In the env, side[0] is the top (strongest). In App.tsx side[last] is the top.
  // We already reversed when building sideInts below, so sideTop = strongest.
  return beats(sideTop, pileTop, -1) ? sideTop : pileTop
}

/**
 * Encode the full 630-dim input vector for the ONNX model.
 * Equivalent to: encode_observation_v2(obs, state, player_id) + mask_f
 */
function encodeObservation(state: InferenceState, playerId: number): Float32Array {
  const me = playerId
  const player = state.players[me]

  // ── Card int lists ────────────────────────────────────────────────────────
  const myHandInts    = player.hand.map(cardToEnvInt)
  const myHandSet     = new Set(myHandInts)
  const pileInts      = state.pile.map(cardToEnvInt)
  // App.tsx pile: [oldest ... newest]; newest = pile top
  const pileTop       = pileInts.length > 0 ? pileInts[pileInts.length - 1] : null
  const pileBodyInts  = pileInts.slice(0, -1)

  // App.tsx side: [oldest ... newest], newest = current top challenger
  // diamonds_env_v2 side: side[0] = top (strongest)
  // So we reverse App.tsx side to get env order
  const sideInts      = [...state.side.map(cardToEnvInt)].reverse()

  const stackInts     = state.deck.map(cardToEnvInt)

  // All captured cards (across all players)
  const capturedInts: number[] = []
  for (const p of state.players) {
    capturedInts.push(...p.captured.map(cardToEnvInt))
  }

  // played_history equivalent: cards not in any known location
  const accounted = new Set([
    ...myHandInts, ...pileInts, ...sideInts, ...stackInts, ...capturedInts,
  ])
  const playedInts: number[] = []
  for (let i = 0; i < NUM_CARDS; i++) {
    if (!accounted.has(i)) playedInts.push(i)
  }

  // Unknown cards: not in my hand, not in accounted visible locations
  const unknownSet = new Set<number>()
  for (let i = 0; i < NUM_CARDS; i++) {
    if (!myHandSet.has(i) && !new Set([...pileInts, ...sideInts, ...capturedInts, ...playedInts]).has(i)) {
      unknownSet.add(i)
    }
  }

  const currentWinning = currentWinningCardInt(pileInts, sideInts, state.lastChallengerId)

  // ── Legal cards (for masking & hand_summary) ─────────────────────────────
  const leadSuitEnvIdx  = state.leadSuit !== null ? (UI_SUIT_TO_ENV_IDX[state.leadSuit] ?? null) : null
  const legalCardInts   = getLegalCardInts(state, playerId)
  const legalCardSet    = new Set(legalCardInts)

  // ── Opponent hand counts ──────────────────────────────────────────────────
  const oppHandCounts: number[] = []
  for (const p of state.players) {
    if (p.id !== me) oppHandCounts.push(p.hand.length)
  }

  // ── Scores ────────────────────────────────────────────────────────────────
  const scorePadded = new Array(MAX_PLAYERS).fill(0)
  for (let i = 0; i < state.gameScores.length && i < MAX_PLAYERS; i++) {
    scorePadded[i] = state.gameScores[i]
  }

  // ── Captured counts ───────────────────────────────────────────────────────
  const myCaptured  = player.captured.length
  const oppCaptured = state.players.reduce((s, p) => p.id !== me ? s + p.captured.length : s, 0)

  // ── Unseen diamonds / jokers ──────────────────────────────────────────────
  let unseenDiamonds = 0, unseenJokers = 0
  for (const cid of unknownSet) {
    if (cid >= 52) unseenJokers++
    else if (Math.floor(cid / 13) === 1) unseenDiamonds++  // suit index 1 = diamonds
  }

  // ── turn_players_remaining ────────────────────────────────────────────────
  const remaining = state.players.filter(p => !p.hasActed).map(p => p.id)

  // ── side_top_player ───────────────────────────────────────────────────────
  const sideTopPlayer = state.lastChallengerId

  // ─────────────────────────────────────────────────────────────────────────
  // BASE ENCODE_OBSERVATION (392 dims)
  // ─────────────────────────────────────────────────────────────────────────
  const vec: number[] = []

  // 6×54 card multi-hots (324)
  vec.push(...multiHot(myHandInts))
  vec.push(...multiHot(playedInts))
  vec.push(...multiHot(pileBodyInts))
  vec.push(...multiHot(sideInts))
  vec.push(...oneHotCard(pileTop))
  vec.push(...oneHotCard(currentWinning))

  // 3×MAX_PLAYERS player one-hots (24)
  vec.push(...oneHotPlayer(sideTopPlayer))
  vec.push(...oneHotPlayer(me))
  vec.push(...oneHotPlayer(0))  // dealer — not tracked; use 0 as default

  // phase one-hot (7)
  vec.push(...phaseOneHot(state.phase))

  // lead suit one-hot (5)
  vec.push(...leadSuitFeatures(state.leadSuit))

  // 32 scalars:
  vec.push(player.hand.length)                              // my hand size (1)
  for (let i = 0; i < MAX_PLAYERS - 1; i++) {              // opp hand sizes padded (7)
    vec.push(i < oppHandCounts.length ? oppHandCounts[i] : 0)
  }
  vec.push(state.deck.length)                              // stack size (1)
  for (let i = 0; i < MAX_PLAYERS; i++) {                  // scores padded (8)
    vec.push(scorePadded[i])
  }
  vec.push(myCaptured)                                     // my captured count (1)
  vec.push(oppCaptured)                                    // opp captured count (1)
  vec.push(1)                                              // round_number (1) — not tracked, default 1
  vec.push(state.turnActionCount)                          // turn_number (1)

  // boolean flags (11)
  vec.push(0)                          // round_end_pending
  vec.push(0)                          // player_who_emptied_hand == me
  vec.push(0)                          // player_who_emptied_hand != me but not null
  vec.push(unseenDiamonds)
  vec.push(unseenJokers)
  vec.push(remaining.length)
  vec.push(state.players[state.currentPlayerIndex]?.id === me ? 1 : 0)
  vec.push(state.deferred ? 1 : 0)
  vec.push(0)                          // suit_switched — not tracked in App.tsx
  vec.push(state.phase === 'VOTING' ? 1 : 0)
  vec.push(state.players.length / MAX_PLAYERS)   // num_players normalised

  if (vec.length !== STATE_DIM) {
    console.error(`[AI] Base obs dim mismatch: got ${vec.length}, expected ${STATE_DIM}`)
  }

  // ─────────────────────────────────────────────────────────────────────────
  // V2 EXTENSION (180 dims → total 572)
  // ─────────────────────────────────────────────────────────────────────────

  const suits = [0, 1, 2, 3]  // clubs=0, diamonds=1, hearts=2, spades=3 (env suit idx)

  function suitOfCard(cid: number): number | null {
    if (cid >= 52) return null
    return Math.floor(cid / 13)
  }
  function rankOfCard(cid: number): number {
    return (cid % 13) + 2
  }
  function highestHeldNaturalRank(suit: number): number {
    let best = 0
    for (const cid of myHandSet) {
      if (cid < 52 && suitOfCard(cid) === suit) {
        best = Math.max(best, rankOfCard(cid))
      }
    }
    return best
  }
  function highestUnknownNaturalRank(suit: number): number {
    let best = 0
    for (const cid of unknownSet) {
      if (cid < 52 && suitOfCard(cid) === suit) {
        best = Math.max(best, rankOfCard(cid))
      }
    }
    return best
  }
  function highestUnknownEffectiveThreat(suit: number): number {
    for (const cid of unknownSet) {
      if (cid >= 52) return 15  // joker is a threat
    }
    return highestUnknownNaturalRank(suit)
  }

  // 1) Exact hand vector (54)
  const handVec = new Array(NUM_CARDS).fill(0)
  for (const cid of myHandSet) handVec[cid] = 1
  vec.push(...handVec)

  // 2) Unknown-card vector (54)
  const unknownVec = new Array(NUM_CARDS).fill(0)
  for (const cid of unknownSet) unknownVec[cid] = 1
  vec.push(...unknownVec)

  // 3) Hand summaries (16)
  vec.push(safeDiv(player.hand.length, 13))
  for (const s of suits) {
    const cnt = [...myHandSet].filter(c => c < 52 && suitOfCard(c) === s).length
    vec.push(safeDiv(cnt, 13))
  }
  const diamondsInHand = [...myHandSet].filter(c => c < 52 && suitOfCard(c) === 1).length
  const jokersInHand   = [...myHandSet].filter(c => c >= 52).length
  vec.push(safeDiv(diamondsInHand, 13))
  vec.push(safeDiv(jokersInHand, 2))
  const highestBySuit = suits.map(s => highestHeldNaturalRank(s))
  for (const h of highestBySuit) vec.push(h / 14)
  for (let i = 0; i < suits.length; i++) {
    const myBest = highestBySuit[i]
    const unknownBest = highestUnknownNaturalRank(suits[i])
    vec.push(myBest > 0 && myBest > unknownBest ? 1 : 0)
  }
  // beat_current_count
  let beatCurrentCount = 0
  if (currentWinning === null) {
    beatCurrentCount = [...legalCardSet].filter(c => c < NUM_CARDS).length
  } else {
    beatCurrentCount = [...legalCardSet].filter(c => c < NUM_CARDS && beats(c, currentWinning, leadSuitEnvIdx ?? -1)).length
  }
  vec.push(safeDiv(beatCurrentCount, 13))

  // 4) Uncertainty summaries (17)
  for (const s of suits) {
    const cnt = [...unknownSet].filter(c => c < 52 && suitOfCard(c) === s).length
    vec.push(safeDiv(cnt, 13))
  }
  for (const s of suits) vec.push(highestUnknownNaturalRank(s) / 14)
  for (const s of suits) vec.push(highestUnknownEffectiveThreat(s) / 15)
  const unknownJokers = [...unknownSet].filter(c => c >= 52).length
  vec.push(safeDiv(unknownJokers, 2))
  for (let i = 0; i < suits.length; i++) {
    const myBest = highestBySuit[i]
    const outrank = [...unknownSet].filter(
      c => c < 52 && suitOfCard(c) === suits[i] && rankOfCard(c) > myBest
    ).length
    vec.push(safeDiv(outrank, 13))
  }

  // 5) Control / survivability (7)
  vec.push(beatCurrentCount > 0 ? 1 : 0)
  vec.push(safeDiv(beatCurrentCount, 13))
  const myBestLead = leadSuitEnvIdx !== null ? highestHeldNaturalRank(leadSuitEnvIdx) : 0
  const highestUnknownLead = leadSuitEnvIdx !== null ? highestUnknownNaturalRank(leadSuitEnvIdx) : 0
  const naturalControlInLead = myBestLead > 0 && myBestLead > highestUnknownLead ? 1 : 0
  vec.push(naturalControlInLead)
  vec.push(unknownJokers > 0 ? 1 : 0)
  // overtake_count: how many unknown cards beat the current winning card
  let overtakeCount = 0
  if (currentWinning !== null && currentWinning < 52) {
    for (const cid of unknownSet) {
      if (cid >= 52 || (cid < 52 && beats(cid, currentWinning, -1))) overtakeCount++
    }
  }
  vec.push(safeDiv(overtakeCount, 15))
  vec.push(jokersInHand > 0 ? 1 : 0)
  vec.push(diamondsInHand > 0 ? 1 : 0)

  // 6) Turn geometry (6)
  let turnPosition = 0, remainingAfter = 0, activeTurnCount = 0
  activeTurnCount = remaining.length
  if (state.phase === 'ACTION' && remaining.includes(me)) {
    const idx = remaining.indexOf(me)
    const denom = Math.max(1, remaining.length - 1)
    turnPosition = safeDiv(idx, denom)
    remainingAfter = remaining.length - idx - 1
  }
  const actedCount = state.players.filter(p => p.hasActed).length
  vec.push(turnPosition)
  vec.push(safeDiv(remainingAfter, MAX_PLAYERS - 1))
  vec.push(safeDiv(actedCount, MAX_PLAYERS))
  vec.push(state.phase === 'ACTION' && remaining.includes(me) && remainingAfter === 0 ? 1 : 0)
  vec.push(safeDiv(activeTurnCount, MAX_PLAYERS))
  const oppOutOfCards = state.players.filter(p => p.id !== me && p.hand.length === 0).length
  vec.push(safeDiv(oppOutOfCards, MAX_PLAYERS - 1))

  // 7) Current contest (8)
  const pileSize = pileInts.length
  const sideSize = sideInts.length
  const contestValue = pileSize + sideSize
  const currentBestNatural = currentWinning !== null && currentWinning < 52
    ? rankOfCard(currentWinning) / 14
    : 0
  const currentBestIsJoker = currentWinning !== null && currentWinning >= 52 ? 1 : 0
  const sideTopIsMe = sideTopPlayer === me ? 1 : 0
  vec.push(safeDiv(sideSize, NUM_CARDS))
  vec.push(safeDiv(pileSize, NUM_CARDS))
  vec.push(safeDiv(contestValue, NUM_CARDS))
  vec.push(currentBestNatural)
  vec.push(currentBestIsJoker)
  vec.push(sideTopIsMe)
  vec.push(state.deferred ? 1 : 0)
  vec.push(0)  // suit_switched — not tracked

  // 8) Stakes / horizon (8)
  const oppHandSizes = state.players.filter(p => p.id !== me).map(p => p.hand.length)
  const avgOppHand = oppHandSizes.length > 0 ? oppHandSizes.reduce((a, b) => a + b, 0) / oppHandSizes.length : 0
  const minOppHand = oppHandSizes.length > 0 ? Math.min(...oppHandSizes) : 0
  const maxOppHand = oppHandSizes.length > 0 ? Math.max(...oppHandSizes) : 0
  vec.push(safeDiv(state.deck.length, NUM_CARDS))
  vec.push(safeDiv(pileSize, NUM_CARDS))
  vec.push(safeDiv(sideSize, NUM_CARDS))
  vec.push(safeDiv(player.hand.length, 13))
  vec.push(safeDiv(avgOppHand, 13))
  vec.push(safeDiv(minOppHand, 13))
  vec.push(safeDiv(maxOppHand, 13))
  vec.push(safeDiv(state.turnActionCount, 20))

  // 9) Score / position (6)
  const numTeams = state.numTeams ?? 1
  const myScoreKey = numTeams > 1
    ? getTeamIndex(me, state.players.length, numTeams)
    : me
  const myScore = state.gameScores[myScoreKey] ?? 0
  const oppScoreKeys = state.gameScores.map((_, k) => k).filter(k => k !== myScoreKey)
  const oppScores = oppScoreKeys.map(k => state.gameScores[k] ?? 0)
  const bestOppScore = oppScores.length > 0 ? Math.max(...oppScores) : 0
  const avgOppScore = oppScores.length > 0 ? oppScores.reduce((a, b) => a + b, 0) / oppScores.length : 0
  const firstPlaceScore = Math.max(...state.gameScores)
  const tgt = Math.max(1, state.targetScore)
  vec.push(safeDiv(myScore, tgt))
  vec.push(safeDiv(bestOppScore, tgt))
  vec.push(safeDiv(avgOppScore, tgt))
  vec.push(safeDiv(myScore - firstPlaceScore, tgt))
  vec.push(safeDiv(myScore - bestOppScore, tgt))
  vec.push(numTeams > 1 ? 1 : 0)

  // 10) Vote context (4)
  const endVotes   = Object.values(state.votes).filter(v => v === 'END').length
  const contVotes  = Object.values(state.votes).filter(v => v === 'KEEP').length
  const endingFavorable = myScore >= bestOppScore ? 1 : 0
  const upside = safeDiv(state.deck.length, NUM_CARDS) * Math.max(0, safeDiv(avgOppHand - player.hand.length, 13) + 0.5)
  vec.push(state.phase === 'VOTING' ? 1 : 0)
  vec.push(safeDiv(endVotes, MAX_PLAYERS))
  vec.push(safeDiv(contVotes, MAX_PLAYERS))
  vec.push(Math.max(0, Math.min(1, 0.5 * endingFavorable + 0.5 * upside)))

  if (vec.length !== STATE_DIM_V2) {
    console.error(`[AI] V2 obs dim mismatch: got ${vec.length}, expected ${STATE_DIM_V2}`)
  }

  // ─────────────────────────────────────────────────────────────────────────
  // MASK (58 dims) — appended to obs to form the 630-dim model input
  // ─────────────────────────────────────────────────────────────────────────
  const mask = buildActionMask(state, playerId)
  vec.push(...mask)

  if (vec.length !== 630) {
    console.error(`[AI] Final obs dim mismatch: got ${vec.length}, expected 630`)
  }

  return new Float32Array(vec)
}


// ─── Action mask builder ──────────────────────────────────────────────────────

function getTeamIndex(playerIdx: number, numPlayers: number, numTeams: number): number {
  if (numTeams <= 1) return playerIdx
  const teamSize = Math.floor(numPlayers / numTeams)
  return Math.floor(playerIdx / teamSize)
}

/**
 * Returns an array of 58 floats (1.0 = legal, 0.0 = illegal).
 * Mirrors diamonds_env_v2.legal_action_mask() translated to App.tsx state.
 */
function buildActionMask(state: InferenceState, playerId: number): number[] {
  const mask = new Array(ACTION_SIZE).fill(0)

  if (state.phase === 'VOTING') {
    if (state.players[state.currentPlayerIndex]?.id === playerId) {
      mask[ACTION_VOTE_CONTINUE] = 1
      mask[ACTION_VOTE_END] = 1
    }
    return mask
  }

  if (state.phase !== 'ACTION') return mask
  if (state.players[state.currentPlayerIndex]?.id !== playerId) return mask

  // Legal card plays
  for (const cid of getLegalCardInts(state, playerId)) {
    if (cid < NUM_CARDS) mask[cid] = 1
  }
  // Draw
  if (state.deck.length > 0) mask[ACTION_DRAW] = 1
  // Pass — always legal in ACTION phase for current player
  mask[ACTION_PASS] = 1

  return mask
}

/**
 * Returns env integer card ids that are legal plays for the given player.
 * Mirrors diamonds_env_v2.get_legal_card_ids() and App.tsx isLegalPlay().
 */
function getLegalCardInts(state: InferenceState, playerId: number): number[] {
  const player = state.players[playerId]
  if (!player) return []

  const pileTop = state.pile.length > 0 ? state.pile[state.pile.length - 1] : null
  const hasLeadSuit = player.hand.some(c => c.suit === state.leadSuit && !c.isJoker)

  const legal: number[] = []
  for (const card of player.hand) {
    if (isLegalPlay(card, state.leadSuit, hasLeadSuit, pileTop)) {
      legal.push(cardToEnvInt(card))
    }
  }
  return legal
}

function isLegalPlay(
  card: Card,
  leadSuit: Suit | null,
  hasLeadSuit: boolean,
  pileTop: Card | null,
): boolean {
  if (card.isJoker) return true
  const isLeadSuit   = card.suit === leadSuit
  const isRankMatch  = pileTop !== null && card.rank === pileTop.rank
  const isDiamond    = card.suit === 'D'

  if (hasLeadSuit) {
    return isLeadSuit || isRankMatch
  } else {
    return isLeadSuit || isRankMatch || isDiamond
  }
}


// ─── Main inference function ──────────────────────────────────────────────────

/**
 * Run inference for a CPU player and return an AIMove.
 *
 * @param playerName   "Lucy", "Barney", etc.
 * @param state        Current App.tsx game state (subset)
 * @param playerId     The player's numeric id (0-based)
 */
export async function getAIMove(
  playerName: string,
  state: InferenceState,
  playerId: number,
): Promise<AIMove> {
  const session  = getSession(playerName)
  const styleId  = STYLE_IDS[playerName]
  if (styleId === undefined) throw new Error(`No style_id for player: ${playerName}`)

  const obsData = encodeObservation(state, playerId)
  const mask    = buildActionMask(state, playerId)

  // Build ONNX tensors
  const obsTensor   = new ort.Tensor('float32', obsData, [1, 630])
  const styleTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(styleId)]), [1])

  // DEBUG — remove after confirming
  const legalCount = mask.filter(m => m === 1).length
  console.log(`[AI] ${playerName} mask: ${legalCount} legal actions, ` +
    `hand size: ${state.players[playerId].hand.length}, ` +
    `phase: ${state.phase}, ` +
    `currentPlayerIndex: ${state.currentPlayerIndex}, ` +
    `playerId: ${playerId}`)

  // Run model
  const results = await session.run({ obs: obsTensor, style_ids: styleTensor })
  const logits  = results['policy_logits'].data as Float32Array  // shape (1, 58)

  // Apply mask: set illegal action logits to -Infinity, then argmax
  let bestAction = ACTION_PASS
  let bestLogit  = -Infinity
  for (let i = 0; i < ACTION_SIZE; i++) {
    if (mask[i] === 1 && logits[i] > bestLogit) {
      bestLogit  = logits[i]
      bestAction = i
    }
  }

  // Convert action integer back to AIMove
  return envActionToAIMove(bestAction, state, playerId)
}

function envActionToAIMove(actionInt: number, state: InferenceState, playerId: number): AIMove {
  if (actionInt === ACTION_DRAW) return { type: 'DRAW' }
  if (actionInt === ACTION_PASS) return { type: 'PASS' }
  if (actionInt === ACTION_VOTE_CONTINUE) return { type: 'VOTE', choice: 'KEEP' }
  if (actionInt === ACTION_VOTE_END)      return { type: 'VOTE', choice: 'END' }

  // Card play: actionInt is the env integer card_id
  const uiCardId = ENV_INT_TO_UI_ID[actionInt]
  if (!uiCardId) {
    console.error(`[AI] Unknown action int: ${actionInt}, falling back to PASS`)
    return { type: 'PASS' }
  }

  // Find the actual Card object in the player's hand that matches this env id
  const player = state.players[playerId]
  const card = player.hand.find(c => UI_ID_TO_ENV_INT[c.id] === actionInt)
  if (!card) {
    console.error(`[AI] Model chose card ${uiCardId} (env ${actionInt}) not in hand. Falling back to PASS.`)
    return { type: 'PASS' }
  }

  return { type: 'PLAY', card }
}


// ─── Convenience: load all players at once ────────────────────────────────────

/**
 * Load all 7 AI player models.
 * modelDir should be the path/URL prefix to your models folder.
 * e.g. '/models' → loads '/models/Lucy_update2800.onnx' etc.
 *
 * Pass an explicit map if your filenames differ from the defaults.
 */
export async function loadAllPlayers(
  modelFiles: Record<string, string>
): Promise<void> {
  await Promise.all(
    Object.entries(modelFiles).map(([name, url]) => loadPlayer(name, url))
  )
}
