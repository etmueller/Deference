/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { loadAllPlayers, getAIMove, type InferenceState } from './aiInference'
import { getQuote, getTieQuote, getTieGameQuote } from './quotes';
import { motion, AnimatePresence } from 'motion/react';
import {
  Trophy,
  RotateCcw,
  User,
  Cpu,
  Layers,
  ArrowDownCircle,
  AlertCircle
} from 'lucide-react';

// --- Types & Constants ---

enum Suit {
  HEARTS = 'H',
  DIAMONDS = 'D',
  CLUBS = 'C',
  SPADES = 'S',
  JOKER = 'J'
}

enum Rank {
  TWO = 2, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN,
  JACK, QUEEN, KING, ACE, JOKER
}

interface Card {
  id: string;
  suit: Suit;
  rank: Rank;
  isJoker: boolean;
}

interface Player {
  id: number;
  name: string;
  hand: Card[];
  captured: Card[];
  isAI: boolean;
  hasActed: boolean;
}

type TurnPhase = 'START' | 'ACTION' | 'END_CHECK' | 'ROUND_OVER' | 'GAME_OVER' | 'VOTING';

interface LogEntry {
  id: number;
  text: string;
  type: 'PLAYER' | 'CPU' | 'SYSTEM';
  timestamp: number;
}

let logIdCounter = 0;

type MessageType = 'info' | 'warning' | 'error' | 'success';

const SUITS = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES];
const RANKS = [
  Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN,
  Rank.EIGHT, Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE
];

const CPU_NAMES = ['Lucy', 'Zane', 'Uncle', 'Barney', 'Fabi', 'Jane', 'Noema'];

// --- Helpers ---

const createDeck = (): Card[] => {
  const deck: Card[] = [];
  SUITS.forEach(suit => {
    RANKS.forEach(rank => {
      deck.push({ id: `${suit}-${rank}`, suit, rank, isJoker: false });
    });
  });
  // Add 2 Jokers
  deck.push({ id: 'JOKER-1', suit: Suit.JOKER, rank: Rank.JOKER, isJoker: true });
  deck.push({ id: 'JOKER-2', suit: Suit.JOKER, rank: Rank.JOKER, isJoker: true });
  return deck;
};

const shuffle = (deck: Card[]): Card[] => {
  const newDeck = [...deck];
  for (let i = newDeck.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [newDeck[i], newDeck[j]] = [newDeck[j], newDeck[i]];
  }
  return newDeck;
};

const getSuitColor = (suit: Suit) => {
  if (suit === Suit.HEARTS || suit === Suit.DIAMONDS) return 'text-red-500';
  if (suit === Suit.JOKER) return 'text-purple-500';
  return 'text-zinc-900';
};

// Suit color for display on a dark background (status bar)
const getSuitColorOnDark = (suit: Suit) => {
  if (suit === Suit.HEARTS || suit === Suit.DIAMONDS) return 'text-red-400';
  if (suit === Suit.JOKER) return 'text-purple-400';
  return 'text-white'; // Clubs and Spades: white so they don't disappear
};

const getSuitIcon = (suit: Suit) => {
  switch (suit) {
    case Suit.HEARTS: return '♥';
    case Suit.DIAMONDS: return '♦';
    case Suit.CLUBS: return '♣';
    case Suit.SPADES: return '♠';
    case Suit.JOKER: return '★';
    default: return '';
  }
};

const getRankLabel = (rank: Rank) => {
  if (rank <= 10) return rank.toString();
  if (rank === Rank.JACK) return 'J';
  if (rank === Rank.QUEEN) return 'Q';
  if (rank === Rank.KING) return 'K';
  if (rank === Rank.ACE) return 'A';
  if (rank === Rank.JOKER) return 'JK';
  return '';
};

// Returns the 0-based team index for a player using sequential (block) assignment.
// e.g. 6 players, 3 teams → P0,P1=T0  P2,P3=T1  P4,P5=T2
const getTeamIndex = (playerIdx: number, totalPlayers: number, totalTeams: number): number => {
  if (totalTeams <= 1) return playerIdx;
  const teamSize = Math.floor(totalPlayers / totalTeams);
  return Math.floor(playerIdx / teamSize);
};

// Normalisation reference used only for AI inference input features.
const calcTargetScore = (numPlayers: number, numTeams: number): number => {
  const units = numTeams > 1 ? numTeams : numPlayers;
  return Math.ceil(100 / units) + 2;
};

// Compute interleaved turn order for team modes.
// Teams are sequential blocks; players are interleaved one-per-team per slot.
// e.g. 4p 2v2: [0,2,1,3]  6p 3v3: [0,3,1,4,2,5]  6p 2v2v2: [0,2,4,1,3,5]
// Re-exported from src/engine/GameEngine.ts — keep in sync.
export const computeInterleavedOrder = (numPlayers: number, numTeams: number): number[] => {
  if (numTeams <= 1) return Array.from({ length: numPlayers }, (_, i) => i);
  const teamSize = Math.floor(numPlayers / numTeams);
  const order: number[] = [];
  for (let slot = 0; slot < teamSize; slot++) {
    for (let t = 0; t < numTeams; t++) {
      order.push(t * teamSize + slot);
    }
  }
  return order;
};

// Return the next player index by cycling through the turnOrder array.
const nextInOrder = (current: number, order: number[]): number => {
  if (order.length === 0) return (current + 1);
  const pos = order.indexOf(current);
  if (pos === -1) return order[0];
  return order[(pos + 1) % order.length];
};

// Available team modes per even player count. Odd counts are always Free for All.
type TeamMode = { label: string; numTeams: number };
const TEAM_MODES: Record<number, TeamMode[]> = {
  4: [
    { label: 'Free for All', numTeams: 1 },
    { label: '2v2',          numTeams: 2 },
  ],
  6: [
    { label: 'Free for All', numTeams: 1 },
    { label: '3v3',          numTeams: 2 },
    { label: '2v2v2',        numTeams: 3 },
  ],
  8: [
    { label: 'Free for All', numTeams: 1 },
    { label: '4v4',          numTeams: 2 },
    { label: '2v2v2v2',      numTeams: 4 },
  ],
};

const ALL_NEGATIVE_QUOTES = [
  "Nobody won that round. Nobody.",
  "The cards giveth. The cards taketh away. Mostly taketh.",
  "A round where everyone lost. Impressive, in a way.",
  "The pile was never worth fighting over. You fought anyway.",
  "Collective punishment. The game has spoken.",
  "Even the winner of that round lost that round.",
  " Get on in there with your shovel, sexton.",
  "Nobody left that round with more than they came in with.",
  "That round was a lesson in the futility of ambition.",
  "The house always wins. In this case, the house is the discard pile.",
  "A perfectly distributed disaster.",
  "When everyone loses, is anyone really losing?",
  "The cards conspired against all of you equally.",
  "Zero sum. Emphasis on zero.",
  "Well. That happened.",
  "We are the anti-Goats!",
  "Are you trying to lose?",
  "That round was brought to you by mutual destruction.",
  "As the nodes change, so must the edges, otherwise the map will lose its grip on the territory.",
];

// --- Setup constants ---

type SkillLevel = 'expert' | 'mid' | 'beginner' | 'human';
type SeatEntry = { name: string; isHuman: boolean };

const ROSTER_PLAYERS: { name: string; isHuman: boolean; skill: SkillLevel }[] = [
  { name: 'You',    isHuman: true,  skill: 'human'    },
  { name: 'Noema',  isHuman: false, skill: 'expert'   },
  { name: 'Jane',   isHuman: false, skill: 'expert'   },
  { name: 'Lucy',   isHuman: false, skill: 'mid'      },
  { name: 'Fabi',   isHuman: false, skill: 'mid'      },
  { name: 'Barney', isHuman: false, skill: 'mid'      },
  { name: 'Uncle',  isHuman: false, skill: 'beginner' },
  { name: 'Zane',   isHuman: false, skill: 'beginner' },
];

const TEAM_COLOR_CLASSES = [
  { border: 'border-blue-500',    bg: 'bg-blue-50',    text: 'text-blue-600',    labelBg: 'bg-blue-500 text-white'    },
  { border: 'border-red-500',     bg: 'bg-red-50',     text: 'text-red-600',     labelBg: 'bg-red-500 text-white'     },
  { border: 'border-emerald-500', bg: 'bg-emerald-50', text: 'text-emerald-600', labelBg: 'bg-emerald-500 text-white' },
  { border: 'border-amber-500',   bg: 'bg-amber-50',   text: 'text-amber-600',   labelBg: 'bg-amber-500 text-white'   },
];

const SkillBadge = ({ skill }: { skill: SkillLevel }) => {
  if (skill === 'human') return <span className="text-[9px] font-black uppercase opacity-50 tracking-wider">HUMAN</span>;
  const stars = skill === 'expert' ? '★★★' : skill === 'mid' ? '★★☆' : '★☆☆';
  const color = skill === 'expert' ? 'text-amber-500' : skill === 'mid' ? 'text-zinc-500' : 'text-zinc-500';
  return <span className={`text-[11px] font-mono leading-none ${color}`}>{stars}</span>;
};

const getDisplayName = (seatsArr: (SeatEntry | null)[], idx: number): string => {
  const s = seatsArr[idx];
  if (!s) return '';
  if (s.isHuman) return 'You';
  const sameCount = seatsArr.slice(0, idx + 1).filter(x => x && x.name === s.name && !x.isHuman).length;
  const totalSame = seatsArr.filter(x => x && x.name === s.name && !x.isHuman).length;
  if (totalSame <= 1) return s.name;
  return `${s.name} ${'ABCDEFG'[sameCount - 1]}`;
};

// --- Components ---

interface CardViewProps {
  key?: React.Key;
  card: Card;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
  isHidden?: boolean;
  size?: 'xs' | 'sm' | 'md' | 'lg';
}

const CardView = ({ card, onClick, disabled, className = "", isHidden = false, size = "md" }: CardViewProps) => {
  const sizeClasses = {
    xs: 'w-10 h-14 sm:w-12 sm:h-18 text-[8px]',
    sm: 'w-12 h-18 sm:w-16 sm:h-24 text-[10px]',
    md: 'w-20 h-28 sm:w-24 sm:h-36 text-xs',
    lg: 'w-28 h-40 sm:w-32 sm:h-48 text-sm'
  };

  const iconSizes = {
    xs: 'text-xl',
    sm: 'text-2xl',
    md: 'text-4xl',
    lg: 'text-6xl'
  };

  return (
    <motion.div
      layoutId={card.id}
      whileHover={!disabled && !isHidden ? { y: -5, scale: 1.05, rotate: 1 } : {}}
      whileTap={!disabled && !isHidden ? { scale: 0.95 } : {}}
      onClick={!disabled ? onClick : undefined}
      className={`relative rounded-xl border-2 flex flex-col items-center justify-center cursor-pointer transition-all shadow-md shrink-0
        ${isHidden ? 'bg-[#141414] border-[#141414]' : 'bg-white border-[#141414]'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-amber-500'}
        ${sizeClasses[size]}
        ${className}`}
    >
      {!isHidden && (
        <>
          <div className={`absolute top-1 left-1.5 font-black ${getSuitColor(card.suit)}`}>
            {getRankLabel(card.rank)}
          </div>
          <div className={`${iconSizes[size]} ${getSuitColor(card.suit)} drop-shadow-sm`}>
            {getSuitIcon(card.suit)}
          </div>
          <div className={`absolute bottom-1 right-1.5 font-black rotate-180 ${getSuitColor(card.suit)}`}>
            {getRankLabel(card.rank)}
          </div>
        </>
      )}
      {isHidden && (
        <div className="w-full h-full flex items-center justify-center p-2">
          <div className="w-full h-full border border-white/10 rounded-lg flex items-center justify-center bg-gradient-to-br from-zinc-800 to-black">
            <Layers size={size === 'sm' ? 16 : 32} className="text-white opacity-30" />
          </div>
        </div>
      )}
    </motion.div>
  );
};

// Module-level guard — lives outside React's render cycle, immune to re-renders
const cpuVotesThisPhase = new Set<number>();

export default function App() {
  // --- AI model loading (runs once on mount) ---
  useEffect(() => {
    loadAllPlayers({
      Lucy:  '/Deference/models/Lucy_1650.onnx',
      Zane:  '/Deference/models/Zane_950.onnx',
      Uncle: '/Deference/models/Uncle_2150.onnx',
      Barney:'/Deference/models/Barney_800.onnx',
      Fabi:  '/Deference/models/Fabi_1925.onnx',
      Jane:  '/Deference/models/Jane_2350.onnx',
      Noema: '/Deference/models/Noema_2350.onnx',
    }).catch(err => console.error('[AI] Model load error:', err))
  }, [])

  // --- State ---
  const [deck, setDeck] = useState<Card[]>([]);
  const [pile, setPile] = useState<Card[]>([]);
  const [side, setSide] = useState<Card[]>([]);
  const [players, setPlayers] = useState<Player[]>([]);
  const [currentPlayerIndex, setCurrentPlayerIndex] = useState(0);
  const [phase, setPhase] = useState<TurnPhase>('START');
  const [leadSuit, setLeadSuit] = useState<Suit | null>(null);
  const [deferred, setDeferred] = useState(false);
  const [gameScores, setGameScores] = useState<number[]>([]);
  const [turnActionCount, setTurnActionCount] = useState(0);
  const [message, setMessage] = useState("");
  const [messageType, setMessageType] = useState<MessageType>('info');
  const [winner, setWinner] = useState<number | null>(null);
  const [lastChallengerId, setLastChallengerId] = useState<number | null>(null);
  const [gameStarted, setGameStarted] = useState(false);
  const [roundLeaderIndex, setRoundLeaderIndex] = useState(0);
  const [turnLeaderIndex, setTurnLeaderIndex] = useState(0);
  const [playerCount, setPlayerCount] = useState(3);
  const [numTeams, setNumTeams] = useState(1); // 1 = Free for All; >1 = team count
  const [numRounds, setNumRounds] = useState(3);
  const roundsPlayedRef = useRef(0);
  const [gameSpeed, setGameSpeed] = useState<'pause' | 'slow' | 'normal' | 'fast'>('normal');
  const [jokerBanner, setJokerBanner] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const [votes, setVotes] = useState<{ [playerId: number]: 'KEEP' | 'END' }>({});
  const roundVoteTriggers = useRef<Set<number>>(new Set());
  const [lastAction, setLastAction] = useState<Record<number, { label: string; red: boolean }>>({});
  const [lastCapture, setLastCapture] = useState<{ playerName: string, count: number } | null>(null);
  const [turnOrder, setTurnOrder] = useState<number[]>([]);
  const [turnQuote, setTurnQuote] = useState<{ name: string; quote: string } | null>(null);
  const turnQuoteTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [roundWinQuote, setRoundWinQuote] = useState<{ name: string; quote: string } | null>(null);
  const [winGameQuote, setWinGameQuote] = useState<string | null>(null);
  const [tieGameQuote, setTieGameQuote] = useState<{ speaker: string; quote: string } | null>(null);

  // --- Setup flow state ---
  const [setupStep, setSetupStep] = useState<1 | 2>(1);
  const [setupSeats, setSetupSeats] = useState<(SeatEntry | null)[]>([]);
  const chosenPlayersRef = useRef<{ name: string; isAI: boolean }[]>([]);
  const seatSlotRefs = useRef<(HTMLDivElement | null)[]>([]);
  const [isDraggingSetup, setIsDraggingSetup] = useState(false);
  const [rulesOpen, setRulesOpen] = useState(false);

  const addLog = useCallback((msg: string, type: 'PLAYER' | 'CPU' | 'SYSTEM' = 'SYSTEM') => {
    setLogs(prev => {
      // Safety dedup: skip if last entry has identical text within the same second
      if (prev.length > 0 && prev[0].text === msg && Date.now() - prev[0].timestamp < 1000) return prev;
      return [{ id: ++logIdCounter, text: msg, type, timestamp: Date.now() }, ...prev].slice(0, 100);
    });
  }, []);

  const showTurnQuote = useCallback((name: string, quote: string) => {
    if (turnQuoteTimer.current) clearTimeout(turnQuoteTimer.current);
    setTurnQuote({ name, quote });
    turnQuoteTimer.current = setTimeout(() => setTurnQuote(null), 5000);
  }, []);

  const showMessage = useCallback((text: string, type: MessageType = 'info') => {
    setMessage(text);
    setMessageType(type);
  }, []);

  const currentPlayer = players[currentPlayerIndex] as Player | undefined;
  
  const liveScores = numTeams > 1
    ? new Array(numTeams).fill(0).map((_, teamIdx) => {
        const teamPlayers = players.filter((_, i) => getTeamIndex(i, players.length, numTeams) === teamIdx);
        const teamRoundScore = teamPlayers.reduce((acc, p) => acc + (p.captured.length - p.hand.length), 0);
        return (gameScores[teamIdx] || 0) + teamRoundScore;
      })
    : players.map((p, i) => (gameScores[i] || 0) + (p.captured.length - p.hand.length));

  const maxScore = liveScores.length > 0 ? Math.max(...liveScores) : 0;
  // Auto-scroll log to top
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = 0;
    }
  }, [logs]);

  // Auto-clear message after 2 seconds
  useEffect(() => {
    if (!message) return;
    const timer = setTimeout(() => setMessage(''), 2000);
    return () => clearTimeout(timer);
  }, [message]);

  // --- Logic ---

  const startRound = useCallback((initConfigs?: { name: string; isAI: boolean }[], initCount?: number) => {
    const newDeck = shuffle(createDeck());

    // Use provided configs (first round) or fall back to the ref set at game start (subsequent rounds)
    const configs = initConfigs ?? chosenPlayersRef.current;
    const count = initCount ?? configs.length;

    const newPlayers: Player[] = configs.map((cfg, i) => ({
      id: i,
      name: cfg.name,
      hand: newDeck.splice(0, 4),
      captured: [],
      isAI: cfg.isAI,
      hasActed: false
    }));

    setDeck(newDeck);
    setPile([]);
    setSide([]);
    setPlayers(newPlayers);
    setVotes({});

    // Only reset scores and round counter if it's the very first round
    if (!gameStarted) {
      if (numTeams > 1) {
        setGameScores(new Array(numTeams).fill(0));
      } else {
        setGameScores(new Array(count).fill(0));
      }
      roundsPlayedRef.current = 0;
    }

    const nextLeader = gameStarted ? (roundLeaderIndex + 1) % count : 0;
    setRoundLeaderIndex(nextLeader);
    setTurnLeaderIndex(nextLeader);
    setCurrentPlayerIndex(nextLeader);

    // Compute interleaved turn order for team modes, rotated to start at the round leader.
    const baseOrder = computeInterleavedOrder(count, numTeams);
    const startPos = baseOrder.indexOf(nextLeader);
    const rotatedOrder = startPos >= 0
      ? [...baseOrder.slice(startPos), ...baseOrder.slice(0, startPos)]
      : baseOrder;
    setTurnOrder(rotatedOrder);

    setPhase('START');
    setDeferred(false);
    setLeadSuit(null);
    setTurnActionCount(0);
    setLastChallengerId(null);
    setLastCapture(null);
    roundVoteTriggers.current = new Set();
    setGameStarted(true);
    setLogs([]);
    addLog("--- NEW ROUND STARTED ---", 'SYSTEM');
    showMessage("Round started.");
  }, [gameStarted, roundLeaderIndex, numTeams, addLog, showMessage]);

  const startTurn = useCallback(() => {
    if (deck.length === 0) {
      addLog("Stack depleted — round over.", 'SYSTEM');
      showMessage("Stack depleted — round over!", 'warning');
      const isGameOver = calculateRoundScores(players, gameScores);
      if (!isGameOver) setPhase('ROUND_OVER');
      return;
    }

    const flipped = deck[0];
    const remainingDeck = deck.slice(1);
    
    setDeck(remainingDeck);
    setPile(prev => [...prev, flipped]);
    setLastChallengerId(null);
    setPlayers(prev => prev.map(p => ({ ...p, hasActed: false })));
    setLastAction({});
    
    if (flipped.isJoker) {
      // Joker flipped from Stack: slow down + show banner, then flipper wins all cards
      setGameSpeed('slow');
      setJokerBanner(true);
      setTimeout(() => setJokerBanner(false), 1500);
      const allCards = [...pile, ...side, flipped];
      addLog(`${currentPlayer?.name} flipped a JOKER from the Stack — wins all ${allCards.length} card${allCards.length !== 1 ? 's' : ''} instantly!`, currentPlayer?.isAI ? 'CPU' : 'PLAYER');
      showMessage(`${currentPlayer?.name} flipped a Joker — wins the pile instantly!`, 'success');
      if (currentPlayer) showTurnQuote(currentPlayer.name, getQuote(currentPlayer.name, 'WIN_TURN'));
      setPlayers(prev => prev.map((p, idx) =>
        idx === currentPlayerIndex ? { ...p, captured: [...p.captured, ...allCards] } : p
      ));
      setPile([]);
      setSide([]);
      setPhase('START');
      // The next player in turn order flips to start the new turn
      setCurrentPlayerIndex(nextInOrder(currentPlayerIndex, turnOrder));
    } else {
      setLeadSuit(flipped.suit);
      setDeferred(false);
      setPhase('ACTION');
      setTurnActionCount(0);
      addLog(`${currentPlayer?.name} flipped ${getRankLabel(flipped.rank)}${getSuitIcon(flipped.suit)}`, currentPlayer?.isAI ? 'CPU' : 'PLAYER');
      showMessage(`Lead suit is ${getSuitIcon(flipped.suit)}. Choose an action.`);
      // TRIGGER 1 — Ten of hearts flipped from stack
      if (flipped.rank === Rank.TEN && flipped.suit === Suit.HEARTS && currentPlayer) {
        const q = "Ten of hearts flips on the pile";
        addLog(q, 'SYSTEM');
        showTurnQuote(currentPlayer.name, q);
      }
    }
  }, [deck, pile, side, currentPlayerIndex, players, currentPlayer, turnOrder, addLog, showMessage]);

  const handleVote = (playerId: number, choice: 'KEEP' | 'END') => {
    // Guard: prevent duplicate votes
    if (votes[playerId] !== undefined) return;

    const currentPlayers = players;
    const currentGameScores = gameScores;

    // Log the vote HERE — outside setVotes so React cannot call it multiple times
    addLog(`${currentPlayers[playerId].name} voted: ${choice === 'KEEP' ? 'CONTINUE' : 'END ROUND'}`, currentPlayers[playerId].isAI ? 'CPU' : 'PLAYER');

    // Compute the new votes from closure — safe because the outer guard prevents re-entry
    const newVotes = { ...votes, [playerId]: choice };

    // Pure state update — no side effects inside the updater
    setVotes(prev => {
      if (prev[playerId] !== undefined) return prev;
      return { ...prev, [playerId]: choice };
    });

    // Handle "all voted" logic outside setVotes
    if (Object.keys(newVotes).length === currentPlayers.length) {
      const keepVotes = Object.values(newVotes).filter(v => v === 'KEEP').length;
      const endVotes = Object.values(newVotes).filter(v => v === 'END').length;

      const keepNames = Object.entries(newVotes).filter(([, v]) => v === 'KEEP').map(([id]) => currentPlayers[Number(id)].name);
      const endNames = Object.entries(newVotes).filter(([, v]) => v === 'END').map(([id]) => currentPlayers[Number(id)].name);

      let outcome: 'KEEP' | 'END';
      let summaryMsg: string;

      if (keepVotes > endVotes) {
        outcome = 'KEEP';
        summaryMsg = `Vote result: CONTINUE ${keepVotes} (${keepNames.join(', ')}) | END ROUND ${endVotes} (${endNames.join(', ')}) → CONTINUE wins`;
      } else if (endVotes > keepVotes) {
        outcome = 'END';
        summaryMsg = `Vote result: CONTINUE ${keepVotes} (${keepNames.join(', ')}) | END ROUND ${endVotes} (${endNames.join(', ')}) → END ROUND wins`;
      } else {
        const flip = Math.random() > 0.5 ? 'KEEP' : 'END';
        outcome = flip;
        summaryMsg = `Vote result: Tied ${keepVotes}-${endVotes} → Coin flip decided: ${flip === 'KEEP' ? 'CONTINUE' : 'END ROUND'}`;
      }

      addLog(summaryMsg, 'SYSTEM');
      addLog(outcome === 'KEEP' ? '--- ROUND CONTINUES ---' : '--- ROUND OVER ---', 'SYSTEM');

      setTimeout(() => {
        if (outcome === 'KEEP') {
          setPhase('START');
          showMessage("Vote: Continue playing! Players with no cards will pass.");
        } else {
          const isGameOver = calculateRoundScores(currentPlayers, currentGameScores);
          if (!isGameOver) setPhase('ROUND_OVER');
        }
      }, 2000);
    }
  };

  const calculateRoundScores = useCallback((currentPlayers: Player[], currentGameScores: number[]) => {
    const roundScores = currentPlayers.map(p => p.captured.length - p.hand.length);
    const allNegativeOrZero = roundScores.every(d => d <= 0);
    let newGameScores: number[] = [];

    if (numTeams > 1) {
      newGameScores = new Array(numTeams).fill(0).map((_, teamIdx) => {
        const teamRound = roundScores
          .filter((_, i) => getTeamIndex(i, currentPlayers.length, numTeams) === teamIdx)
          .reduce((a, b) => a + b, 0);
        return currentGameScores[teamIdx] + teamRound;
      });

      addLog("--- ROUND OVER ---");
      newGameScores.forEach((score, i) => {
        const teamRound = score - currentGameScores[i];
        addLog(`Team ${i + 1}: ${teamRound} pts (Total: ${score})`);
      });

      if (allNegativeOrZero) {
        const q = ALL_NEGATIVE_QUOTES[Math.floor(Math.random() * ALL_NEGATIVE_QUOTES.length)];
        addLog(`"${q}"`, 'SYSTEM');
        setRoundWinQuote({ name: '', quote: q });
      } else {
        // WIN_ROUND quote for team mode
        const teamRoundDeltas = newGameScores.map((s, i) => s - currentGameScores[i]);
        const maxTeamRound = Math.max(...teamRoundDeltas);
        const winningTeams = teamRoundDeltas.reduce((acc, d, i) => d === maxTeamRound ? [...acc, i] : acc, [] as number[]);
        if (winningTeams.length === 1) {
          const teamMembers = currentPlayers.filter((_, i) => getTeamIndex(i, currentPlayers.length, numTeams) === winningTeams[0]);
          const speaker = teamMembers[Math.floor(Math.random() * teamMembers.length)];
          if (speaker) {
            const q = getQuote(speaker.name, 'WIN_ROUND');
            addLog(`${speaker.name}: "${q}"`);
            setRoundWinQuote({ name: speaker.name, quote: q });
          }
        } else {
          const q = getTieQuote();
          addLog(`"${q}"`);
          setRoundWinQuote({ name: '', quote: q });
        }
      }
    } else {
      newGameScores = currentGameScores.map((s, i) => s + (roundScores[i] || 0));
      addLog("--- ROUND OVER ---");
      currentPlayers.forEach((p, i) => addLog(`${p.name}: ${roundScores[i]} pts (Total: ${newGameScores[i]})`));

      if (allNegativeOrZero) {
        const q = ALL_NEGATIVE_QUOTES[Math.floor(Math.random() * ALL_NEGATIVE_QUOTES.length)];
        addLog(`"${q}"`, 'SYSTEM');
        setRoundWinQuote({ name: '', quote: q });
      } else {
        // WIN_ROUND quote for FFA mode
        const maxRound = Math.max(...roundScores);
        const roundWinners = roundScores.reduce((acc, s, i) => s === maxRound ? [...acc, i] : acc, [] as number[]);
        if (roundWinners.length === 1) {
          const rw = currentPlayers[roundWinners[0]];
          const q = getQuote(rw.name, 'WIN_ROUND');
          addLog(`${rw.name}: "${q}"`);
          setRoundWinQuote({ name: rw.name, quote: q });
        } else {
          const q = getTieQuote();
          addLog(`"${q}"`);
          setRoundWinQuote({ name: '', quote: q });
        }
      }
    }

    setGameScores(newGameScores);

    const newRoundsPlayed = roundsPlayedRef.current + 1;
    roundsPlayedRef.current = newRoundsPlayed;

    if (newRoundsPlayed >= numRounds) {
      setPhase('GAME_OVER');
      const maxScore = Math.max(...newGameScores);
      const winnerIndices = newGameScores.reduce((acc: number[], s, i) => s === maxScore ? [...acc, i] : acc, []);
      const isTie = winnerIndices.length > 1;

      if (isTie) {
        addLog(`GAME OVER! Tie between ${winnerIndices.map(i => numTeams > 1 ? `Team ${i + 1}` : currentPlayers[i].name).join(' and ')}!`);
        setWinner(null);
        setTieGameQuote(getTieGameQuote(currentPlayers));
      } else {
        const winnerIdx = winnerIndices[0];
        setWinner(winnerIdx);
        if (numTeams > 1) {
          addLog(`GAME OVER! Team ${winnerIdx + 1} wins!`);
          const teamMembers = currentPlayers.filter((_, i) => getTeamIndex(i, currentPlayers.length, numTeams) === winnerIdx);
          const speaker = teamMembers[Math.floor(Math.random() * teamMembers.length)];
          if (speaker) setWinGameQuote(getQuote(speaker.name, 'WIN_GAME'));
        } else {
          addLog(`GAME OVER! ${currentPlayers[winnerIdx].name} wins!`);
          setWinGameQuote(getQuote(currentPlayers[winnerIdx].name, 'WIN_GAME'));
        }
      }
      return true; // Game is over
    }
    return false; // Game continues
  }, [numTeams, numRounds, addLog]);

  const handleAction = (action: 'PLAY' | 'DRAW' | 'PASS', card?: Card) => {
    if (phase !== 'ACTION') return;

    let turnEndedByJoker = false;

    if (action === 'DRAW') {
      if (deck.length > 0) {
        const drawn = deck[0];
        setDeck(deck.slice(1));
        setPlayers(prev => prev.map((p, idx) =>
          idx === currentPlayerIndex ? { ...p, hand: [...p.hand, drawn], hasActed: true } : p
        ));
        setLastAction(prev => ({ ...prev, [currentPlayerIndex]: { label: 'drew', red: false } }));
        addLog(`${currentPlayer?.name} drew a card.`);
        showMessage(`${currentPlayer?.name} drew a card.`);
      } else {
        showMessage("Stack is empty. Cannot draw.", 'warning');
        return;
      }
    } else if (action === 'PASS') {
      setPlayers(prev => prev.map((p, idx) =>
        idx === currentPlayerIndex ? { ...p, hasActed: true } : p
      ));
      setLastAction(prev => ({ ...prev, [currentPlayerIndex]: { label: 'pass', red: false } }));
      if (currentPlayer?.hand.length === 0) {
        addLog(`${currentPlayer?.name} has no cards — passing.`, currentPlayer?.isAI ? 'CPU' : 'PLAYER');
      } else {
        addLog(`${currentPlayer?.name} passed.`);
      }
      showMessage(`${currentPlayer?.name} passed.`);
    } else if (action === 'PLAY' && card) {
      const pileTop = pile.length > 0 ? pile[pile.length - 1] : null;
      const hasLeadSuit = currentPlayer?.hand.some(c => c.suit === leadSuit && !c.isJoker);
      
      const isJoker = card.isJoker;
      const isLeadSuit = card.suit === leadSuit;
      const isRankMatch = pileTop && card.rank === pileTop.rank;
      const isDiamond = card.suit === Suit.DIAMONDS;
      const isLegalDiamond = isDiamond && !hasLeadSuit;

      // 1. Basic Legality Check
      if (!isJoker && !isLeadSuit && !isRankMatch && !isLegalDiamond) {
        showMessage(`Illegal play! Must play ${getSuitIcon(leadSuit!)}, a rank match, a Joker, or a Diamond (if no ${getSuitIcon(leadSuit!)})`, 'error');
        return;
      }

      // 2. Specific "Must Follow Suit" check for Diamonds
      if (isDiamond && hasLeadSuit && !isLeadSuit && !isRankMatch) {
        showMessage(`Cannot play Diamond while you hold ${getSuitIcon(leadSuit!)}`, 'error');
        return;
      }

      if (isJoker) {
        setSide(prev => [...prev, card]);
        setLastChallengerId(currentPlayerIndex);
        setPlayers(prev => prev.map((p, idx) =>
          idx === currentPlayerIndex ? { ...p, hand: p.hand.filter(c => c.id !== card.id), hasActed: true } : p
        ));
        setLastAction(prev => ({ ...prev, [currentPlayerIndex]: { label: 'JKR', red: false } }));
        addLog(`${currentPlayer?.name} played Joker!`, currentPlayer?.isAI ? 'CPU' : 'PLAYER');
        showMessage(`${currentPlayer?.name} played a Joker — wins the pile!`, 'success');
        // TRIGGER 5 — Joker played
        if (currentPlayer) {
          const q = "But we are all still dead afraid if a Joker should be played";
          addLog(q, 'SYSTEM');
          showTurnQuote(currentPlayer.name, q);
        }
        turnEndedByJoker = true;
      } else if (isRankMatch) {
        // Suit Switch: Merge side into pile and reset challengers
        setPile(prev => [...prev, ...side, card]);
        setSide([]);
        setLeadSuit(card.suit);
        setDeferred(false);
        setLastChallengerId(null);
        setPlayers(prev => prev.map((p, idx) =>
          idx === currentPlayerIndex ? { ...p, hand: p.hand.filter(c => c.id !== card.id), hasActed: true } : p
        ));
        setLastAction(prev => ({ ...prev, [currentPlayerIndex]: { label: getRankLabel(card.rank) + getSuitIcon(card.suit), red: card.suit === Suit.HEARTS || card.suit === Suit.DIAMONDS } }));
        addLog(`${currentPlayer?.name} matched rank ${getRankLabel(card.rank)} and Suit Switched to ${getSuitIcon(card.suit)}`, currentPlayer?.isAI ? 'CPU' : 'PLAYER');
        showMessage(`${currentPlayer?.name} Suit Switched to ${getSuitIcon(card.suit)}!`);
        // TRIGGER 3 — Ten of spades suit switch
        if (card.rank === Rank.TEN && card.suit === Suit.SPADES && currentPlayer) {
          const q = "The ten of spades is played. The leading suit is changed";
          addLog(q, 'SYSTEM');
          showTurnQuote(currentPlayer.name, q);
        }
      } else {
        // Lead suit play or Diamond Defer
        if (isDiamond && !isLeadSuit && !deferred) {
          setDeferred(true);
          addLog(`${currentPlayer?.name} deferred the pile.`, currentPlayer?.isAI ? 'CPU' : 'PLAYER');
          showMessage(`${currentPlayer?.name} deferred the pile!`, 'warning');
          // TRIGGER 4 — Diamond defer
          if (currentPlayer) {
            const opts = [
              "Follow suit, if suit you hold. If you can't, a diamond's bold",
              "Play a diamond if you prefer. Then the pile will be deferred",
              "The pile builds. The stakes grow higher. It's coming down to the wire.",
              "Play a diamond — lock the fight. No one wins the pile tonight",
              "Knowing when to step aside. Knowing when to let it ride",
              "Knowing when a moment is still too small to claim",
              "The clever play is not to take the pile",
            ];
            const q = opts[Math.floor(Math.random() * opts.length)];
            addLog(q, 'SYSTEM');
            showTurnQuote(currentPlayer.name, q);
          }
        }

        const sideTop = side.length > 0 ? side[side.length - 1] : null;
        
        // Determine the current winning card correctly
        // The winning card is either the last side card that beat the pile, or the pile top.
        let currentWinningCard = pileTop;
        if (sideTop && lastChallengerId !== null) {
          currentWinningCard = sideTop;
        }

        // Hierarchy: Joker > Lead Suit > Diamond > Pile
        let beatsCurrent = false;
        if (isJoker) {
          beatsCurrent = true;
        } else if (currentWinningCard?.isJoker) {
          beatsCurrent = false;
        } else if (isLeadSuit) {
          // Lead suit beats non-lead suit, or higher rank of same suit
          if (currentWinningCard?.suit !== leadSuit || card.rank > currentWinningCard.rank) {
            beatsCurrent = true;
          }
        } else if (isDiamond) {
          // Diamond beats non-lead, non-diamond suit, or higher rank diamond
          if (currentWinningCard?.suit !== leadSuit && (currentWinningCard?.suit !== Suit.DIAMONDS || card.rank > currentWinningCard.rank)) {
            beatsCurrent = true;
          }
        }

        if (beatsCurrent) {
          setSide(prev => [...prev, card]);
          setLastChallengerId(currentPlayerIndex);
          const beatsMsg = deferred ? "" : " (Beats Pile)";
          addLog(`${currentPlayer?.name} played ${getRankLabel(card.rank)}${getSuitIcon(card.suit)}${beatsMsg}`, currentPlayer?.isAI ? 'CPU' : 'PLAYER');
        } else {
          setSide(prev => [card, ...prev]);
          addLog(`${currentPlayer?.name} played ${getRankLabel(card.rank)}${getSuitIcon(card.suit)}`, currentPlayer?.isAI ? 'CPU' : 'PLAYER');
        }

        setPlayers(prev => prev.map((p, idx) =>
          idx === currentPlayerIndex ? { ...p, hand: p.hand.filter(c => c.id !== card.id), hasActed: true } : p
        ));
        setLastAction(prev => ({ ...prev, [currentPlayerIndex]: { label: getRankLabel(card.rank) + getSuitIcon(card.suit), red: card.suit === Suit.HEARTS || card.suit === Suit.DIAMONDS } }));
      }
    }

    const nextActionCount = turnActionCount + 1;
    setTurnActionCount(nextActionCount);

    if (turnEndedByJoker || nextActionCount >= players.length) {
      // Use a small delay to ensure state updates are processed or just pass values
      setTimeout(() => {
        setPhase('END_CHECK');
      }, 100);
    } else {
      setCurrentPlayerIndex(nextInOrder(currentPlayerIndex, turnOrder));
    }
  };

  // Resolution effect to avoid closure issues
  useEffect(() => {
    if (phase === 'END_CHECK') {
      resolveTurn();
    }
  }, [phase]);

  const resolveTurn = () => {
    const sideTop = side.length > 0 ? side[side.length - 1] : null;
    
    // If someone beat the pile, they win. 
    // BUT: If the pile was deferred with a diamond, it can't be won on that turn (except by joker)
    const isJokerWin = sideTop && sideTop.isJoker;
    const canWin = lastChallengerId !== null && (!deferred || isJokerWin);
    
    let updatedPlayers = [...players];
    let pileWon = false;

    if (canWin && lastChallengerId !== null) {
      addLog(`${players[lastChallengerId].name} won the pile (${pile.length + side.length} cards).`);
      showMessage(`${players[lastChallengerId].name} won the pile!`, 'success');
      // TRIGGER 2 — Jack wins the pile (sideTop is the winning card)
      if (sideTop && sideTop.rank === Rank.JACK) {
        const q = "Jack is played with a smile";
        addLog(q, 'SYSTEM');
        showTurnQuote(players[lastChallengerId].name, q);
      } else {
        showTurnQuote(players[lastChallengerId].name, getQuote(players[lastChallengerId].name, 'WIN_TURN'));
      }
      const allCards = [...pile, ...side];
      setLastCapture({ playerName: players[lastChallengerId].name, count: allCards.length });
      
      updatedPlayers = players.map((p, idx) => 
        idx === lastChallengerId ? { ...p, captured: [...p.captured, ...allCards] } : p
      );
      setPlayers(updatedPlayers);
      setPile([]);
      setSide([]);
      setDeferred(false);
      setTurnLeaderIndex(lastChallengerId);
      setCurrentPlayerIndex(lastChallengerId);
      pileWon = true;
    } else {
      if (deferred) {
        addLog("Pile deferred. Cards merged into pile.");
        showMessage("Pile deferred — cards merge into the pile.", 'warning');
      } else {
        addLog("Pile not won. Cards merged.");
        showMessage("Pile not won — side cards merge into the pile.");
      }
      const newPile = [...pile, ...side];
      setPile(newPile);
      setSide([]);
      setDeferred(false);
      const nextTurnLeader = nextInOrder(turnLeaderIndex, turnOrder);
      setTurnLeaderIndex(nextTurnLeader);
      setCurrentPlayerIndex(nextTurnLeader);
    }

    // Check for Round End at the end of the TURN
    const stackEmpty = deck.length === 0;
    const newlyOut = updatedPlayers.filter(p => p.hand.length === 0 && !roundVoteTriggers.current.has(p.id));

    if (stackEmpty) {
      // Stack empty — this was the last turn, auto-end, no vote
      addLog("Stack depleted — round over.", 'SYSTEM');
      const isGameOver = calculateRoundScores(updatedPlayers, gameScores);
      if (!isGameOver) setPhase('ROUND_OVER');
    } else if (newlyOut.length > 0) {
      // First time this player hits 0 cards — trigger vote
      newlyOut.forEach(p => roundVoteTriggers.current.add(p.id));
      const outPlayer = newlyOut[0];
      addLog(`${outPlayer.name} is out of cards — vote: Continue or End Round?`, 'SYSTEM');
      showMessage(`${outPlayer.name} is out of cards — vote!`, 'warning');
      setVotes({});
      setPhase('VOTING');
    } else {
      setPhase('START');
    }
  };

  const resetGame = useCallback(() => {
    setDeck([]);
    setPile([]);
    setSide([]);
    setPlayers([]);
    setCurrentPlayerIndex(0);
    setPhase('START');
    setLeadSuit(null);
    setDeferred(false);
    setGameScores([]);
    setTurnActionCount(0);
    setMessage('');
    setMessageType('info');
    setWinner(null);
    setTieGameQuote(null);
    setLastChallengerId(null);
    setGameStarted(false);
    setRoundLeaderIndex(0);
    setTurnLeaderIndex(0);
    setLogs([]);
    setVotes({});
    setLastCapture(null);
    setTurnOrder([]);
    setSetupStep(1);
    setSetupSeats([]);
    chosenPlayersRef.current = [];
    setGameSpeed('normal');
  }, []);

  // CPU voting effect — module-level Set guards against re-render duplicates.
  useEffect(() => {
    if (phase !== 'VOTING') return
    cpuVotesThisPhase.clear()
    const timer = setTimeout(async () => {
      // Collect each CPU's choice locally so we can resolve without the stale
      // `votes` closure (handleVote reads the closure value, not live state).
      const roundVotes: Record<number, 'KEEP' | 'END'> = {};

      for (const p of players) {
        if (!p.isAI || cpuVotesThisPhase.has(p.id)) continue
        cpuVotesThisPhase.add(p.id)

        const inferState: InferenceState = {
          players, pile, side, deck, leadSuit, deferred,
          currentPlayerIndex: p.id,
          turnOrder, turnLeaderIndex, lastChallengerId,
          turnActionCount, gameScores, targetScore: calcTargetScore(playerCount, numTeams), numTeams, phase, votes,
        }

        let choice: 'KEEP' | 'END';
        try {
          const move = await getAIMove(p.name, inferState, p.id)
          choice = move.type === 'VOTE' ? move.choice : 'KEEP'
        } catch {
          choice = Math.random() > 0.5 ? 'KEEP' : 'END'
        }
        handleVote(p.id, choice)
        roundVotes[p.id] = choice
      }

      // Spectator mode: no human player means handleVote's "all voted" check
      // never fires (it builds newVotes from the stale empty-object closure).
      // Resolve the vote here directly once all CPUs have cast their ballot.
      const hasHuman = players.some(p => !p.isAI);
      if (!hasHuman && Object.keys(roundVotes).length === players.length) {
        const keepVotes = Object.values(roundVotes).filter(v => v === 'KEEP').length;
        const endVotes  = Object.values(roundVotes).filter(v => v === 'END').length;

        const keepNames = Object.entries(roundVotes).filter(([, v]) => v === 'KEEP').map(([id]) => players[Number(id)]?.name ?? '');
        const endNames  = Object.entries(roundVotes).filter(([, v]) => v === 'END').map(([id]) => players[Number(id)]?.name ?? '');

        let outcome: 'KEEP' | 'END';
        let summaryMsg: string;
        if (keepVotes > endVotes) {
          outcome = 'KEEP';
          summaryMsg = `Vote result: CONTINUE ${keepVotes} (${keepNames.join(', ')}) | END ROUND ${endVotes} (${endNames.join(', ')}) → CONTINUE wins`;
        } else if (endVotes > keepVotes) {
          outcome = 'END';
          summaryMsg = `Vote result: CONTINUE ${keepVotes} (${keepNames.join(', ')}) | END ROUND ${endVotes} (${endNames.join(', ')}) → END ROUND wins`;
        } else {
          const flip = Math.random() > 0.5 ? 'KEEP' : 'END';
          outcome = flip;
          summaryMsg = `Vote result: Tied ${keepVotes}-${endVotes} → Coin flip decided: ${flip === 'KEEP' ? 'CONTINUE' : 'END ROUND'}`;
        }

        addLog(summaryMsg, 'SYSTEM');
        addLog(outcome === 'KEEP' ? '--- ROUND CONTINUES ---' : '--- ROUND OVER ---', 'SYSTEM');

        setTimeout(() => {
          if (outcome === 'KEEP') {
            setPhase('START');
            showMessage("Vote: Continue playing! Players with no cards will pass.");
          } else {
            const isGameOver = calculateRoundScores(players, gameScores);
            if (!isGameOver) setPhase('ROUND_OVER');
          }
        }, 2000);
      }
    }, 1000)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase])

  // Notify when any player with 0 cards reaches ACTION phase — they may draw or pass.
  useEffect(() => {
    if (phase !== 'ACTION' || !currentPlayer || currentPlayer.hand.length > 0) return;
    addLog(`${currentPlayer.name} has no cards — may draw or pass.`, currentPlayer.isAI ? 'CPU' : 'PLAYER');
    if (!currentPlayer.isAI) showMessage('You have no cards — draw from the stack or pass.');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, currentPlayerIndex]);

  // AI game-action effect (START / ACTION phases).
  useEffect(() => {
    if (phase === 'VOTING') return; // handled by the effect above
    if (gameSpeed === 'pause') return; // paused — wait for user to resume

    const startDelay  = gameSpeed === 'slow' ? 2000 : gameSpeed === 'fast' ? 200 : 1000;
    const actionDelay = gameSpeed === 'slow' ? 2000 : gameSpeed === 'fast' ? 200 : 1500;

    if (currentPlayer?.isAI) {
      if (phase === 'START') {
        const timer = setTimeout(() => {
          startTurn();
        }, startDelay);
        return () => clearTimeout(timer);
      }

      if (phase === 'ACTION') {
        const timer = setTimeout(async () => {
          if (!currentPlayer?.isAI || !currentPlayer.name) return

          // Build the InferenceState subset that aiInference.ts needs
          const inferState: InferenceState = {
            players,
            pile,
            side,
            deck,
            leadSuit,
            deferred,
            currentPlayerIndex,
            turnOrder,
            turnLeaderIndex,
            lastChallengerId,
            turnActionCount,
            gameScores,
            targetScore: calcTargetScore(playerCount, numTeams),
            numTeams,
            phase,
            votes,
          }

          try {
            const move = await getAIMove(currentPlayer.name, inferState, currentPlayer.id)
            if (move.type === 'PLAY') {
              handleAction('PLAY', move.card as Card)
            } else if (move.type === 'DRAW') {
              handleAction('DRAW')
            } else {
              handleAction('PASS')
            }
          } catch (err) {
            console.error(`[AI] ${currentPlayer.name} inference failed, falling back to PASS:`, err)
            handleAction('PASS')
          }
        }, actionDelay)
        return () => clearTimeout(timer)
      }
    }
  }, [phase, currentPlayerIndex, leadSuit, pile, deck.length, startTurn, currentPlayer?.isAI, gameSpeed]);

  // --- Render Helpers ---

  const sideTop = side.length > 0 ? side[side.length - 1] : null;
  const pileTop = pile.length > 0 ? pile[pile.length - 1] : null;

  // Human player may be at any seat index (or absent in spectator mode)
  const humanPlayerIdx = players.findIndex(p => !p.isAI);
  const isSpectator = humanPlayerIdx === -1;
  const humanPlayer = isSpectator ? undefined : players[humanPlayerIdx];

  // Adaptive scoreboard sizing based on player count
  const sbConfig = playerCount <= 2
    ? { scoreSz: 'text-3xl',     nameSz: 'text-sm',     pad: 'px-3 py-3',     statSz: 'text-[15px]', useGrid: false }
    : playerCount <= 4
    ? { scoreSz: 'text-[28px]',  nameSz: 'text-sm',     pad: 'px-2 py-2.5',   statSz: 'text-[14px]', useGrid: false }
    : playerCount <= 5
    ? { scoreSz: 'text-2xl',     nameSz: 'text-xs',     pad: 'px-2 py-2',     statSz: 'text-[13px]', useGrid: false }
    : { scoreSz: 'text-xl',      nameSz: 'text-[10px]', pad: 'px-1.5 py-1.5', statSz: 'text-[12px]', useGrid: true  };

  return (
    <div className="h-screen bg-[#E4E3E0] text-[#141414] font-sans selection:bg-[#141414] selection:text-[#E4E3E0] flex flex-col overflow-hidden">
      {/* Header - Minimal */}
      <header className="border-b border-[#141414] px-4 py-2 flex justify-between items-center bg-white/30 shrink-0">
        <div className="flex items-baseline gap-2">
          <h1 className="text-xl font-bold tracking-tighter uppercase italic font-serif">DEFERENCE</h1>
          <p className="text-[10px] uppercase tracking-widest opacity-60 font-bold">Diamonds are for the clever.</p>
        </div>
        <div className="flex gap-3 items-center">
          {gameStarted && (
            <div className="flex items-center gap-1">
              {(['pause', 'slow', 'normal', 'fast'] as const).map(s => (
                <button
                  key={s}
                  onClick={() => setGameSpeed(s)}
                  title={s === 'pause' ? 'Pause' : s === 'slow' ? 'Slow (2s)' : s === 'normal' ? 'Normal' : 'Fast (0.2s)'}
                  className={`w-7 h-7 flex items-center justify-center text-[13px] border border-[#141414] transition-colors
                    ${gameSpeed === s ? 'bg-[#141414] text-[#E4E3E0]' : 'hover:bg-zinc-100'}`}
                >
                  {s === 'pause' ? '⏸' : s === 'slow' ? '🐢' : s === 'normal' ? '▶' : '⚡'}
                </button>
              ))}
            </div>
          )}
          <div className="text-right flex items-center gap-2">
            <p className="text-[10px] uppercase opacity-60 font-bold">Round</p>
            <p className="font-mono font-bold text-lg">{roundsPlayedRef.current + 1}/{numRounds}</p>
          </div>
          <button
            onClick={resetGame}
            className="p-1 hover:bg-[#141414] hover:text-[#E4E3E0] transition-colors border border-[#141414] rounded"
          >
            <RotateCcw size={14} />
          </button>
        </div>
      </header>

      <main className="flex-1 grid grid-cols-[65%_35%] grid-rows-[auto_1fr] overflow-hidden">
        {/* TOP-LEFT: Status bar + Notification bar + Card zones */}
        <div className="border-r border-b border-[#141414] flex flex-col overflow-hidden">
          {/* Status Bar */}
          <div className="bg-[#141414] text-[#E4E3E0] px-4 py-2 flex items-center justify-between text-xs font-mono shrink-0">
            <div className="flex gap-6 items-center">
              <span className="flex items-center gap-2">
                <span className="opacity-50 uppercase">Lead:</span>
                <span className={`font-bold text-[22px] leading-none ${leadSuit ? getSuitColorOnDark(leadSuit) : (pile.length > 0 ? getSuitColorOnDark(pile[pile.length - 1].suit) : 'opacity-30')}`}>
                  {leadSuit ? getSuitIcon(leadSuit) : (pile.length > 0 ? getSuitIcon(pile[pile.length - 1].suit) : '∅')}
                </span>
              </span>
              <span className="flex items-center gap-2 border-l border-white/20 pl-6">
                <span className="opacity-50 uppercase">Deferred:</span>
                <span className={`font-bold ${deferred ? 'text-amber-400' : ''}`}>{deferred ? 'YES' : 'NO'}</span>
              </span>
              <span className="flex items-center gap-2 border-l border-white/20 pl-6">
                <span className="opacity-50 uppercase">Stack:</span>
                <span className="font-bold">{deck.length}</span>
              </span>
            </div>
            <div className="flex items-center gap-2 bg-amber-500 text-[#141414] px-3 py-0.5 rounded-full font-bold">
              {currentPlayer?.isAI ? <Cpu size={12} /> : <User size={12} />}
              {currentPlayer?.isAI ? `${currentPlayer.name}'S` : 'YOUR'} TURN
            </div>
          </div>

          {/* Message Bar — fixed height, always rendered to prevent layout shift */}
          <div className={`h-8 px-4 flex items-center text-xs font-bold shrink-0 transition-colors duration-150
            ${message
              ? messageType === 'error'   ? 'bg-red-600 text-white'
              : messageType === 'warning' ? 'bg-amber-400 text-[#141414]'
              : messageType === 'success' ? 'bg-green-600 text-white'
              : 'bg-[#141414]/80 text-[#E4E3E0]'
              : 'bg-transparent'}`}
          >
            {message || '\u00A0'}
          </div>

          {/* Middle: Card Zones — fixed height, flush against status bar */}
          <div className="shrink-0 flex items-stretch justify-around px-2 py-2 gap-2">
            {/* Stack */}
            <div className="flex-1 h-[156px] border border-[#141414] border-dashed rounded-xl flex flex-col items-center justify-center bg-white/10 relative">
              <span className="absolute top-1 left-2 text-[9px] uppercase font-black opacity-30">Stack</span>
              {deck.length > 0 ? (
                <div 
                  onClick={phase === 'START' && !currentPlayer?.isAI ? startTurn : undefined}
                  className={`relative cursor-pointer transition-transform ${phase === 'START' && !currentPlayer?.isAI ? 'hover:scale-105' : 'opacity-50'}`}
                >
                  <CardView card={deck[0]} isHidden size="md" />
                  <div className="absolute -bottom-2 -right-2 bg-[#141414] text-[#E4E3E0] px-2 py-0.5 text-xs font-mono font-bold rounded">
                    {deck.length}
                  </div>
                </div>
              ) : <div className="text-[10px] opacity-20 italic">Empty</div>}
            </div>

            {/* Pile */}
            <div className={`flex-1 h-[156px] border border-[#141414] rounded-xl flex flex-col items-center justify-center bg-white/20 relative ${deferred ? 'ring-2 ring-amber-500' : ''}`}>
              <span className="absolute top-1 left-2 text-[9px] uppercase font-black opacity-30">Pile</span>
              {pile.length > 0 ? (
                <div className="relative">
                  <CardView card={pile[pile.length - 1]} size="md" />
                  <div className="absolute -top-2 -right-2 bg-red-600 text-white w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold border border-white">
                    {pile.length}
                  </div>
                </div>
              ) : <div className="text-[10px] opacity-20 italic">Empty</div>}
            </div>

            {/* Side */}
            <div className="flex-1 h-[156px] border border-[#141414] rounded-xl flex flex-col items-center justify-center bg-white/30 relative">
              <span className="absolute top-1 left-2 text-[9px] uppercase font-black opacity-30">Side</span>
              {side.length > 0 ? (
                <div className="relative">
                  <CardView card={side[side.length - 1]} size="md" />
                  {side.length > 1 && (
                    <div className="absolute -bottom-2 -right-2 bg-[#141414] text-white px-1.5 py-0.5 text-[9px] rounded">
                      +{side.length - 1}
                    </div>
                  )}
                </div>
              ) : <div className="text-[10px] opacity-20 italic">Empty</div>}
              <div className="absolute bottom-1 left-0 right-0 text-center text-[9px] font-bold">
                {lastChallengerId !== null
                  ? <span className="text-amber-600">Leading: {players[lastChallengerId]?.name}</span>
                  : <span className="opacity-30">No winner yet</span>}
              </div>
            </div>
          </div>

          {/* Turn quote — appears below card zones, fades after 3s */}
          <AnimatePresence>
            {turnQuote && (
              <motion.div
                key={turnQuote}
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="shrink-0 px-4 pb-1 text-center"
              >
                <span className="text-amber-500 font-serif text-[20px] leading-snug">
                  <span className="not-italic font-black text-amber-600">{turnQuote.name}: </span>
                  <span className="italic">"{turnQuote.quote}"</span>
                </span>
              </motion.div>
            )}
          </AnimatePresence>

        </div>

        {/* BOTTOM-LEFT: Your Hand + Buttons */}
        <div className="border-r border-[#141414] flex flex-col p-2 bg-white/40 row-start-2 col-start-1">
            <div className="flex items-center gap-2 mb-1">
              <ArrowDownCircle size={14} className="text-amber-600" />
              <span className="text-[10px] uppercase font-black">{isSpectator ? 'Spectator Mode' : 'Your Hand'}</span>
            </div>

            {isSpectator ? (
              <div className="flex-1 flex items-center justify-center text-[10px] uppercase font-black opacity-20 tracking-widest">
                CPU controls all moves
              </div>
            ) : (
              <>
                {/* Mobile: 3-col grid. Desktop: single row. */}
                <div className="py-1">
                  <div className="grid grid-cols-3 gap-1 md:hidden">
                    <AnimatePresence>
                      {players.length > 0 && humanPlayer && humanPlayer.hand.map((card) => (
                        <div key={card.id} className="flex justify-center">
                          <CardView
                            card={card}
                            size="sm"
                            onClick={() => handleAction('PLAY', card)}
                            disabled={phase !== 'ACTION' || currentPlayer?.isAI || currentPlayerIndex !== humanPlayerIdx}
                          />
                        </div>
                      ))}
                    </AnimatePresence>
                  </div>
                  <div className="hidden md:flex justify-center items-center gap-1 overflow-x-hidden">
                    <AnimatePresence>
                      {players.length > 0 && humanPlayer && humanPlayer.hand.map((card) => (
                        <CardView
                          key={card.id}
                          card={card}
                          size={humanPlayer.hand.length > 8 ? "xs" : humanPlayer.hand.length > 5 ? "sm" : "md"}
                          onClick={() => handleAction('PLAY', card)}
                          disabled={phase !== 'ACTION' || currentPlayer?.isAI || currentPlayerIndex !== humanPlayerIdx}
                        />
                      ))}
                    </AnimatePresence>
                  </div>
                  {players.length > 0 && humanPlayer && humanPlayer.hand.length === 0 && phase !== 'VOTING' && (
                    <div className="flex items-center justify-center gap-2 opacity-20 italic text-xs py-1">
                      <AlertCircle size={14} /> No cards
                    </div>
                  )}
                </div>

                {/* Actions — stacked full-width on mobile, side by side on desktop */}
                <div className="flex flex-col md:flex-row justify-center gap-2 mt-2">
                  <button
                    disabled={phase !== 'ACTION' || currentPlayer?.isAI || currentPlayerIndex !== humanPlayerIdx}
                    onClick={() => handleAction('DRAW')}
                    className="w-full md:flex-1 md:max-w-[100px] py-2 md:py-1.5 border border-[#141414] text-[10px] font-black uppercase hover:bg-[#141414] hover:text-[#E4E3E0] disabled:opacity-30 transition-all rounded shadow-[2px_2px_0px_0px_#141414] active:translate-x-[1px] active:translate-y-[1px] active:shadow-none bg-white"
                  >
                    Draw
                  </button>
                  <button
                    disabled={phase !== 'ACTION' || currentPlayer?.isAI || currentPlayerIndex !== humanPlayerIdx}
                    onClick={() => handleAction('PASS')}
                    className="w-full md:flex-1 md:max-w-[100px] py-2 md:py-1.5 border border-[#141414] text-[10px] font-black uppercase hover:bg-[#141414] hover:text-[#E4E3E0] disabled:opacity-30 transition-all rounded shadow-[2px_2px_0px_0px_#141414] active:translate-x-[1px] active:translate-y-[1px] active:shadow-none bg-white"
                  >
                    Pass
                  </button>
                </div>
              </>
            )}
          </div>

        {/* TOP-RIGHT: Scoreboard — row 1, col 2; height matches top-left via grid row */}
        <section className="border-b border-[#141414] flex flex-col overflow-hidden bg-white/50 row-start-1 col-start-2">
            <div className="px-4 py-2 border-b border-[#141414] bg-[#141414] text-[#E4E3E0] flex justify-between items-center shrink-0">
              <h2 className="text-[10px] uppercase font-black flex items-center gap-2 italic">
                <Trophy size={12} className="text-amber-400" /> Scoreboard
              </h2>
            </div>
            <div className="flex-1 overflow-hidden p-1.5">
              {numTeams > 1 ? (
                <div className={sbConfig.useGrid ? 'grid grid-cols-2 gap-1 h-full content-start' : 'space-y-1.5 h-full'}>
                  {new Array(numTeams).fill(0).map((_, teamIdx) => {
                    const teamPlayers = players.filter((_, i) => getTeamIndex(i, players.length, numTeams) === teamIdx);
                    const score = liveScores[teamIdx] || 0;
                    const isLeader = score === maxScore && maxScore > 0;
                    const isCurrentTeam = getTeamIndex(currentPlayerIndex, players.length, numTeams) === teamIdx;

                    return (
                      <div
                        key={teamIdx}
                        className={`border rounded-lg overflow-hidden transition-all duration-300 bg-white
                          ${isLeader ? 'border-l-4 border-l-amber-500 ring-1 ring-amber-400' :
                            isCurrentTeam ? 'border-l-4 border-l-blue-500' : 'border-[#141414]/10'}
                        `}
                      >
                        {/* Team header — score is most prominent */}
                        <div className={`flex justify-between items-center ${sbConfig.pad}
                          ${isLeader ? 'bg-amber-50' : isCurrentTeam ? 'bg-blue-50/60' : ''}`}
                        >
                          <div className="flex items-center gap-1.5 min-w-0">
                            <span className={`${sbConfig.nameSz} font-black italic uppercase truncate ${isLeader ? 'text-amber-700' : ''}`}>
                              Team {teamIdx + 1}
                            </span>
                          </div>
                          <span className={`font-mono font-black shrink-0 ml-1 ${sbConfig.scoreSz} ${isLeader ? 'text-amber-600' : score < 0 ? 'text-red-600' : 'text-[#141414]'}`}>
                            {score}
                          </span>
                        </div>
                        {/* Individual player sub-rows */}
                        <div className="divide-y divide-[#141414]/5 border-t border-[#141414]/10">
                          {teamPlayers.map(p => (
                            <div
                              key={p.id}
                              className={`flex justify-between items-center px-2 py-1 ${sbConfig.statSz} transition-colors
                                ${currentPlayerIndex === p.id ? 'bg-blue-500 text-white' : ''}`}
                            >
                              <div className="flex items-center gap-0.5">
                                {p.isAI ? <Cpu size={7} /> : <User size={7} />}
                                <span className="font-bold truncate max-w-[40px]">{p.name}</span>
                                {p.hasActed && (
                                  <span className="flex items-center gap-0.5">
                                    <span className={`font-black text-[11px] ${currentPlayerIndex === p.id ? 'text-white' : 'text-green-500'}`}>✓</span>
                                    {lastAction[p.id] && <span className={`font-bold text-[10px] ${lastAction[p.id].red ? 'text-red-500' : (currentPlayerIndex === p.id ? 'text-white/70' : 'text-[#141414]')}`}>{lastAction[p.id].label}</span>}
                                  </span>
                                )}
                              </div>
                              <div className="flex gap-2 font-mono font-bold">
                                <span className={currentPlayerIndex === p.id ? 'text-white' : 'text-[#141414]'}>✋{p.hand.length}</span>
                                <span className={currentPlayerIndex === p.id ? 'text-white/70' : 'text-zinc-500'}>+{p.captured.length}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className={sbConfig.useGrid ? 'grid grid-cols-2 gap-1 h-full content-start' : 'space-y-1.5 h-full'}>
                  {players.map((p, i) => {
                    const score = liveScores[i] || 0;
                    const isLeader = score === maxScore && maxScore > 0;
                    const isCurrent = currentPlayerIndex === i;

                    return (
                      <div
                        key={p.id}
                        className={`${sbConfig.pad} border rounded-lg flex justify-between items-center transition-all duration-300
                          ${isCurrent ? 'border-l-4 border-l-blue-500 bg-blue-50/50' : 'bg-white border-[#141414]/10'}
                          ${isLeader ? 'border-l-4 border-l-amber-500 ring-1 ring-amber-400' : ''}
                        `}
                      >
                        <div className="flex flex-col min-w-0">
                          <div className="flex items-center gap-1">
                            <span className={`${sbConfig.nameSz} font-black uppercase italic flex items-center gap-0.5 truncate ${isLeader ? 'text-amber-700' : ''}`}>
                              {p.isAI ? <Cpu size={8} /> : <User size={8} />} {p.name}
                            </span>
                            {p.hasActed && (
                              <span className="shrink-0 flex items-center gap-0.5">
                                <span className="text-green-500 font-black text-[11px]">✓</span>
                                {lastAction[p.id] && <span className={`font-bold ${sbConfig.statSz} ${lastAction[p.id].red ? 'text-red-500' : 'text-[#141414]'}`}>{lastAction[p.id].label}</span>}
                              </span>
                            )}
                          </div>
                          <div className={`flex gap-2 mt-0.5 font-mono font-bold ${sbConfig.statSz}`}>
                            <span className="text-[#141414]">✋{p.hand.length}</span>
                            <span className="text-zinc-500">+{p.captured.length}</span>
                          </div>
                        </div>
                        <span className={`font-mono font-black shrink-0 ml-1 ${sbConfig.scoreSz} ${isLeader ? 'text-amber-600' : score < 0 ? 'text-red-600' : 'text-[#141414]'}`}>
                          {score}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
            
            {/* Round Progress Footer */}
            <div className="px-3 py-1.5 bg-[#141414] text-[9px] text-white/80 flex justify-between items-center border-t border-white/10 shrink-0 font-mono">
              <div className="flex items-center gap-2">
                <span className="text-amber-400 font-black uppercase">Stakes:</span>
                <span>{pile.length} cards</span>
              </div>
              {lastCapture && (
                <div className="flex items-center gap-2">
                  <span className="text-blue-400 font-black uppercase">Last:</span>
                  <span className="truncate max-w-[70px]">{lastCapture.playerName} (+{lastCapture.count})</span>
                </div>
              )}
            </div>
          </section>

        {/* BOTTOM-RIGHT: Game Log — row 2, col 2 */}
        <section className="flex flex-col overflow-hidden bg-white row-start-2 col-start-2">
            <div className="px-4 py-2 border-b border-[#141414] bg-[#141414] text-[#E4E3E0] shrink-0">
              <h2 className="text-[10px] uppercase font-black flex items-center gap-2 italic">
                <Layers size={12} className="text-blue-400" /> Game Log
              </h2>
            </div>
            <div 
              ref={logContainerRef}
              className="flex-1 overflow-y-auto p-4 font-mono text-sm space-y-2 custom-scrollbar bg-[#F8F8F8]"
            >
              <AnimatePresence initial={false}>
                {logs.map((log) => (
                  <motion.div
                    key={log.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className={`pb-1 border-b border-[#141414]/5 last:border-0 leading-tight
                      ${log.type === 'PLAYER' ? 'text-blue-600 font-bold' : 
                        log.type === 'CPU' ? 'text-red-600 italic' : 
                        'text-zinc-500'}
                      ${log.text.startsWith('---') ? 'bg-amber-50 p-2 rounded text-amber-800 font-black text-center my-2 text-xs' : ''}
                    `}
                  >
                    <span className="opacity-30 text-[9px] mr-2">[{new Date(log.timestamp).toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' })}]</span>
                    <span className="font-black mr-1 text-[10px]">
                      {log.type === 'PLAYER' ? 'YOU:' : log.type === 'CPU' ? 'CPU:' : 'SYS:'}
                    </span>
                    {log.text}
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
        </section>
      </main>

      {/* Overlays */}
      <AnimatePresence>
        {jokerBanner && (
          <motion.div
            key="joker-banner"
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.04 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-[70] flex items-center justify-center pointer-events-none"
          >
            <div className="bg-[#141414] text-[#E4E3E0] px-14 py-10 text-center border-2 border-amber-400 shadow-[8px_8px_0px_0px_rgba(251,191,36,0.4)]">
              <p className="text-6xl mb-4">🃏</p>
              <p className="text-2xl font-serif italic font-bold text-amber-400 tracking-wide">A Joker appears...</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {phase === 'VOTING' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-[#141414]/85 backdrop-blur-md z-[60] flex items-center justify-center p-4"
          >
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05, duration: 0.25 }}
              className="w-full max-w-4xl border-2 border-[#141414] bg-[#E4E3E0] shadow-[10px_10px_0px_0px_#141414] flex flex-col overflow-hidden max-h-[90vh]"
            >
              {/* Header */}
              <div className="bg-[#141414] text-[#E4E3E0] px-5 py-3 flex items-center gap-3 shrink-0">
                <Layers size={18} className="text-amber-400 shrink-0" />
                <div>
                  <h2 className="text-base font-serif italic font-black uppercase tracking-tighter leading-none">Player Out of Cards</h2>
                  <p className="text-[10px] opacity-50 uppercase tracking-widest mt-0.5">Vote to continue or end the round</p>
                </div>
              </div>

              {/* Body — two columns */}
              <div className="flex flex-col md:flex-row flex-1 overflow-hidden min-h-0">

                {/* LEFT: Info panel */}
                <div className="flex-1 flex flex-col overflow-hidden border-b md:border-b-0 md:border-r border-[#141414]/20">

                  {/* Stakes row */}
                  <div className="flex gap-0 border-b border-[#141414]/15 shrink-0">
                    <div className="flex-1 px-4 py-3 border-r border-[#141414]/15">
                      <p className="text-[9px] uppercase font-black opacity-40 tracking-widest mb-0.5">Cards at Stake</p>
                      <p className="font-mono font-black text-2xl">{pile.length + side.length}</p>
                      <p className="text-[9px] opacity-40 mt-0.5">pile + side</p>
                    </div>
                    <div className="flex-1 px-4 py-3">
                      <p className="text-[9px] uppercase font-black opacity-40 tracking-widest mb-0.5">Stack Remaining</p>
                      <p className="font-mono font-black text-2xl">{deck.length}</p>
                      <p className="text-[9px] opacity-40 mt-0.5">cards left to flip</p>
                    </div>
                  </div>

                  {/* Player hand sizes + scores */}
                  <div className="border-b border-[#141414]/15 shrink-0 px-4 py-3">
                    <p className="text-[9px] uppercase font-black opacity-40 tracking-widest mb-2">Players</p>
                    <div className="space-y-1">
                      {numTeams > 1
                        ? new Array(numTeams).fill(0).map((_, teamIdx) => {
                            const teamPlayers = players.filter((_, i) => getTeamIndex(i, players.length, numTeams) === teamIdx);
                            const tc = TEAM_COLOR_CLASSES[teamIdx % TEAM_COLOR_CLASSES.length];
                            return (
                              <div key={teamIdx} className={`border-l-2 ${tc.border} pl-2`}>
                                <p className={`text-[9px] font-black uppercase ${tc.text} mb-0.5`}>Team {teamIdx + 1} — {liveScores[teamIdx] ?? 0} pts</p>
                                {teamPlayers.map(p => (
                                  <div key={p.id} className="flex items-center justify-between text-xs font-mono py-0.5">
                                    <span className="flex items-center gap-1 opacity-70">
                                      {p.isAI ? <Cpu size={9} /> : <User size={9} />}
                                      <span className="font-bold uppercase text-[10px]">{p.name}</span>
                                      {p.hand.length === 0 && <span className="text-amber-600 font-black text-[9px]">OUT</span>}
                                    </span>
                                    <span className="font-black">{p.hand.length} cards</span>
                                  </div>
                                ))}
                              </div>
                            );
                          })
                        : players.map((p, i) => (
                            <div key={p.id} className="flex items-center justify-between">
                              <span className="flex items-center gap-1.5 text-[11px] font-bold uppercase">
                                {p.isAI ? <Cpu size={9} /> : <User size={9} />}
                                {p.name}
                                {p.hand.length === 0 && <span className="text-amber-600 font-black text-[9px] ml-1">OUT</span>}
                              </span>
                              <div className="flex items-center gap-3 font-mono text-[11px]">
                                <span className="opacity-50">{p.hand.length} cards</span>
                                <span className="font-black w-10 text-right">{liveScores[i] ?? 0} pts</span>
                              </div>
                            </div>
                          ))
                      }
                    </div>
                  </div>

                  {/* CPU vote status */}
                  <div className="border-b border-[#141414]/15 shrink-0 px-4 py-3">
                    <p className="text-[9px] uppercase font-black opacity-40 tracking-widest mb-2">Votes Cast</p>
                    <div className="flex flex-wrap gap-1.5">
                      {players.map(p => {
                        const v = votes[p.id];
                        return (
                          <div key={p.id} className={`flex items-center gap-1.5 px-2 py-1 border text-[9px] font-bold rounded
                            ${v === 'KEEP' ? 'bg-[#141414] text-[#E4E3E0] border-[#141414]'
                            : v === 'END'  ? 'bg-red-600 text-white border-red-700'
                            : 'bg-white border-[#141414]/30 opacity-40'}`}>
                            {p.isAI ? <Cpu size={8} /> : <User size={8} />}
                            <span className="uppercase">{p.name}</span>
                            <span className="opacity-70">
                              {v === 'KEEP' ? '→ CONTINUE' : v === 'END' ? '→ END' : '…'}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Game log */}
                  <div className="flex-1 overflow-hidden flex flex-col min-h-0 px-4 py-3">
                    <p className="text-[9px] uppercase font-black opacity-40 tracking-widest mb-2 shrink-0">Recent Log</p>
                    <div className="flex-1 overflow-y-auto font-mono text-[10px] space-y-1 custom-scrollbar">
                      {logs.slice(0, 15).map(log => (
                        <div key={log.id} className={`leading-tight
                          ${log.type === 'PLAYER' ? 'text-blue-600 font-bold' :
                            log.type === 'CPU'    ? 'text-red-600 italic' :
                            'text-zinc-500'}
                          ${log.text.startsWith('---') ? 'text-amber-700 font-black' : ''}`}>
                          {log.text}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* RIGHT: Explanation + buttons */}
                <div className="w-full md:w-72 shrink-0 flex flex-col p-5 gap-4 bg-white/40">

                  {/* Option explanations */}
                  <div className="space-y-3">
                    <div className="border border-[#141414]/20 rounded p-3 bg-white">
                      <p className="text-[9px] uppercase font-black tracking-widest mb-1 text-red-600">End Round</p>
                      <p className="text-[11px] leading-snug opacity-70">
                        Score now. Each player earns <span className="font-bold text-[#141414]">captured − cards in hand</span>.
                        Player with most points wins the round.
                      </p>
                    </div>
                    <div className="border border-[#141414]/20 rounded p-3 bg-white">
                      <p className="text-[9px] uppercase font-black tracking-widest mb-1 text-green-700">Continue Playing</p>
                      <p className="text-[11px] leading-snug opacity-70">
                        Keep going. The player with no cards will <span className="font-bold text-[#141414]">draw or pass</span> each turn.
                        More cards can be captured before scoring.
                      </p>
                    </div>
                  </div>

                  {/* Buttons / spectator notice */}
                  <div className="mt-auto space-y-2">
                    {isSpectator ? (
                      <div className="border border-[#141414]/20 rounded p-3 bg-white text-center">
                        <p className="text-[10px] uppercase font-black opacity-40 tracking-widest">CPU players deciding…</p>
                      </div>
                    ) : (
                      <>
                        <p className="text-[9px] uppercase font-black opacity-40 tracking-widest">Your Vote</p>
                        <button
                          disabled={votes[humanPlayerIdx] !== undefined}
                          onClick={() => handleVote(humanPlayerIdx, 'END')}
                          className={`w-full py-3.5 border-2 border-[#141414] font-black uppercase text-sm tracking-widest
                            shadow-[4px_4px_0px_0px_#141414] active:translate-x-[2px] active:translate-y-[2px] active:shadow-none
                            transition-all rounded
                            ${votes[humanPlayerIdx] === 'END' ? 'bg-[#141414] text-[#E4E3E0]' : 'bg-white hover:bg-red-50'}`}
                        >
                          End Round
                          <span className="block text-[8px] font-bold opacity-50 normal-case tracking-normal mt-0.5">Score what's captured now</span>
                        </button>
                        <button
                          disabled={votes[humanPlayerIdx] !== undefined}
                          onClick={() => handleVote(humanPlayerIdx, 'KEEP')}
                          className={`w-full py-3.5 border-2 border-[#141414] font-black uppercase text-sm tracking-widest
                            shadow-[4px_4px_0px_0px_#141414] active:translate-x-[2px] active:translate-y-[2px] active:shadow-none
                            transition-all rounded
                            ${votes[humanPlayerIdx] === 'KEEP' ? 'bg-[#141414] text-[#E4E3E0]' : 'bg-white hover:bg-amber-50'}`}
                        >
                          Continue
                          <span className="block text-[8px] font-bold opacity-50 normal-case tracking-normal mt-0.5">Keep playing for more cards</span>
                        </button>
                        {votes[humanPlayerIdx] !== undefined && (
                          <p className="text-[9px] text-center opacity-40 font-bold uppercase tracking-widest">
                            Voted — waiting for others…
                          </p>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
        {!gameStarted && (() => {
          // --- Setup helpers (inline to access state) ---
          const findSeatAtPoint = (x: number, y: number): number =>
            seatSlotRefs.current.findIndex(ref => {
              if (!ref) return false;
              const r = ref.getBoundingClientRect();
              return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
            });

          const handleRosterDrop = (player: { name: string; isHuman: boolean }, px: number, py: number) => {
            const target = findSeatAtPoint(px, py);
            if (target < 0) return;
            setSetupSeats(prev => {
              const next = [...prev];
              if (player.isHuman) next.forEach((s, i) => { if (s?.isHuman) next[i] = null; });
              next[target] = { name: player.name, isHuman: player.isHuman };
              return next;
            });
          };

          const handleSeatDrop = (fromIdx: number, px: number, py: number) => {
            const target = findSeatAtPoint(px, py);
            setSetupSeats(prev => {
              const next = [...prev];
              const dragged = next[fromIdx];
              if (!dragged) return prev;
              if (target < 0) {
                next[fromIdx] = null; // dragged to roster area → clear
              } else if (target !== fromIdx) {
                const tmp = next[target];
                next[target] = dragged;
                next[fromIdx] = tmp; // swap
              }
              return next;
            });
          };

          const handleRandomize = () => {
            const cpus = ROSTER_PLAYERS.filter(p => !p.isHuman);
            const shuffled = [...cpus].sort(() => Math.random() - 0.5);
            const humanSeat = Math.floor(Math.random() * playerCount);
            const next: (SeatEntry | null)[] = Array.from({ length: playerCount }, (_, i) => {
              if (i === humanSeat) return { name: 'You', isHuman: true };
              const cpuIdx = i < humanSeat ? i : i - 1;
              return { name: shuffled[cpuIdx % shuffled.length].name, isHuman: false };
            });
            setSetupSeats(next);
          };

          const handleStart = () => {
            const configs = setupSeats.map(s => ({ name: s!.name, isAI: !s!.isHuman }));
            chosenPlayersRef.current = configs;
            setPlayerCount(setupSeats.length);
            startRound(configs, setupSeats.length);
          };

          const allFilled = setupSeats.length === playerCount && setupSeats.every(Boolean);
          const humanCount = setupSeats.filter(s => s?.isHuman).length;
          const canStart = allFilled && humanCount <= 1;

          // Seat card shown inside a seat slot
          const SeatCard = ({ seatIdx }: { seatIdx: number }) => {
            const seat = setupSeats[seatIdx];
            if (!seat) return null;
            const displayName = getDisplayName(setupSeats, seatIdx);
            const rp = ROSTER_PLAYERS.find(r => r.name === seat.name);
            return (
              <motion.div
                drag
                dragSnapToOrigin
                dragElastic={0.25}
                whileDrag={{ scale: 1.08, zIndex: 200, opacity: 0.85 }}
                onDragStart={() => setIsDraggingSetup(true)}
                onDragEnd={(_, info) => { setIsDraggingSetup(false); handleSeatDrop(seatIdx, info.point.x, info.point.y); }}
                className={`w-full flex items-center justify-between gap-1 px-2 py-1.5 cursor-grab active:cursor-grabbing
                  ${seat.isHuman ? 'bg-[#141414] text-[#E4E3E0]' : 'bg-white text-[#141414]'}
                  border border-[#141414] rounded select-none`}
              >
                <div className="flex flex-col min-w-0">
                  <span className="text-[11px] font-black uppercase tracking-tight truncate leading-tight">{displayName}</span>
                  <SkillBadge skill={rp?.skill ?? 'mid'} />
                </div>
                <button
                  onPointerDown={e => e.stopPropagation()}
                  onClick={() => setSetupSeats(prev => { const n = [...prev]; n[seatIdx] = null; return n; })}
                  className={`shrink-0 text-[14px] leading-none opacity-40 hover:opacity-100 transition-opacity ml-1
                    ${seat.isHuman ? 'text-[#E4E3E0]' : 'text-[#141414]'}`}
                >
                  ×
                </button>
              </motion.div>
            );
          };

          return (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-[#E4E3E0]/95 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            >
              <div className={`w-full border-2 border-[#141414] bg-white shadow-[10px_10px_0px_0px_#141414] flex flex-col overflow-hidden
                ${setupStep === 1 ? 'max-w-md' : 'max-w-5xl h-[90vh]'}`}>

                {/* Header */}
                <div className="border-b border-[#141414] px-5 py-3 flex items-center justify-between bg-[#141414] text-[#E4E3E0] shrink-0">
                  <div>
                    <h2 className="text-lg font-serif italic font-bold tracking-tighter">DEFERENCE</h2>
                    <p className="text-[9px] text-red-500 uppercase tracking-widest font-bold">Diamonds are for the clever.</p>
                  </div>
                  <span className="text-[9px] uppercase font-black opacity-40 tracking-widest">Step {setupStep} / 2</span>
                </div>

                {setupStep === 1 ? (
                  /* ── STEP 1: Game Mode ── */
                  <div className="p-8 flex flex-col gap-6">
                    <div className="space-y-5">
                      {/* Player count */}
                      <div>
                        <label className="text-[10px] uppercase font-bold opacity-50 block mb-2">Number of Players</label>
                        <div className="flex gap-1.5">
                          {[2, 3, 4, 5, 6, 7, 8].map(n => (
                            <button
                              key={n}
                              onClick={() => { setPlayerCount(n); setNumTeams(1); }}
                              className={`flex-1 py-2 border border-[#141414] text-xs font-mono font-bold
                                ${playerCount === n ? 'bg-[#141414] text-[#E4E3E0]' : 'hover:bg-zinc-100'}`}
                            >{n}</button>
                          ))}
                        </div>
                      </div>

                      {/* Game mode */}
                      {TEAM_MODES[playerCount] && (
                        <div>
                          <label className="text-[10px] uppercase font-bold opacity-50 block mb-2">Game Mode</label>
                          <div className="flex gap-1.5 flex-wrap">
                            {TEAM_MODES[playerCount].map(mode => (
                              <button
                                key={mode.numTeams}
                                onClick={() => { setNumTeams(mode.numTeams); }}
                                className={`flex-1 py-2 border border-[#141414] text-xs font-mono font-bold
                                  ${numTeams === mode.numTeams ? 'bg-[#141414] text-[#E4E3E0]' : 'hover:bg-zinc-100'}`}
                              >{mode.label}</button>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Number of rounds */}
                      <div>
                        <label className="text-[10px] uppercase font-bold opacity-50 block mb-2">Number of Rounds</label>
                        <div className="flex items-center gap-4">
                          <button
                            onClick={() => setNumRounds((r: number) => Math.max(1, r - 1))}
                            className="w-10 h-10 border border-[#141414] text-lg font-bold hover:bg-zinc-100 transition-colors"
                          >−</button>
                          <span className="font-mono font-bold text-xl flex-1 text-center">{numRounds}</span>
                          <button
                            onClick={() => setNumRounds((r: number) => Math.min(10, r + 1))}
                            className="w-10 h-10 border border-[#141414] text-lg font-bold hover:bg-zinc-100 transition-colors"
                          >+</button>
                        </div>
                      </div>
                    </div>

                    {/* Collapsible rules */}
                    <div className="border border-[#141414]/20 rounded">
                      <button
                        onClick={() => setRulesOpen(o => !o)}
                        className="w-full flex justify-between items-center px-3 py-2 text-[10px] font-black uppercase tracking-widest hover:bg-zinc-50 transition-colors"
                      >
                        <span>How to Play</span>
                        <span className="opacity-50">{rulesOpen ? '▲' : '▼'}</span>
                      </button>
                      <AnimatePresence initial={false}>
                        {rulesOpen && (
                          <motion.div
                            key="rules"
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.2 }}
                            className="overflow-hidden"
                          >
                            <div className="px-3 pb-4 pt-1 space-y-3 border-t border-[#141414]/10 text-left">
                              {[
                                {
                                  heading: 'Objective',
                                  body: 'Have the most points after all rounds are played.\nScore = cards captured − cards left in hand at round end.',
                                },
                                {
                                  heading: 'Each Turn',
                                  body: 'The active player flips a card from the stack — this sets the lead suit. Then each player must play a card, draw, or pass.',
                                },
                                {
                                  heading: 'Playing a Card',
                                  body: 'Follow lead suit, with exceptions:\n• No lead suit? Play a Diamond (defers the pile) or match the rank (suit switch).\n• Joker beats everything and wins the pile instantly.',
                                },
                                {
                                  heading: 'Winning the Pile',
                                  body: 'The highest lead-suit card wins. If the pile was deferred with a Diamond, only a Joker can win it that turn.',
                                },
                                {
                                  heading: 'Voting',
                                  body: 'When a player runs out of cards, everyone votes:\nCONTINUE — keep playing, empty-handed player draws or passes.\nEND ROUND — score immediately.',
                                },
                              ].map(({ heading, body }) => (
                                <div key={heading}>
                                  <p className="text-[9px] font-black uppercase tracking-widest mb-0.5">{heading}</p>
                                  <p className="text-[11px] opacity-60 leading-snug whitespace-pre-line">{body}</p>
                                </div>
                              ))}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    <button
                      onClick={() => { setSetupSeats(new Array(playerCount).fill(null)); setSetupStep(2); }}
                      className="w-full py-4 bg-[#141414] text-[#E4E3E0] font-bold uppercase tracking-widest hover:bg-zinc-800 transition-colors"
                    >
                      Continue →
                    </button>
                  </div>
                ) : (
                  /* ── STEP 2: Player Selection ── */
                  <div className="flex-1 flex overflow-hidden">

                    {/* Left panel: roster */}
                    <div className="w-44 shrink-0 border-r border-[#141414] flex flex-col overflow-hidden">
                      <div className="px-3 pt-3 pb-1 shrink-0">
                        <p className="text-[9px] uppercase font-black text-[#141414] tracking-widest">Roster</p>
                        <p className="text-[8px] text-gray-600 mt-0.5">Drag to a seat →</p>
                      </div>
                      <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
                        {ROSTER_PLAYERS.map(player => (
                          <motion.div
                            key={player.name}
                            drag
                            dragSnapToOrigin
                            dragElastic={0.25}
                            whileDrag={{ scale: 1.08, zIndex: 200, opacity: 0.85 }}
                            onDragStart={() => setIsDraggingSetup(true)}
                            onDragEnd={(_, info) => { setIsDraggingSetup(false); handleRosterDrop(player, info.point.x, info.point.y); }}
                            className={`flex items-center justify-between gap-2 px-2.5 py-2 border rounded
                              cursor-grab active:cursor-grabbing select-none
                              ${player.isHuman
                                ? 'bg-[#141414] text-[#E4E3E0] border-[#141414]'
                                : 'bg-white hover:bg-zinc-50 border-gray-800 text-[#141414]'}`}
                          >
                            <div className="flex flex-col min-w-0">
                              <span className="text-[11px] font-black uppercase tracking-tight truncate leading-tight">
                                {player.isHuman ? 'You' : player.name}
                              </span>
                              <SkillBadge skill={player.skill} />
                            </div>
                            {player.isHuman
                              ? <User size={11} className="shrink-0 opacity-60" />
                              : <Cpu size={11} className="shrink-0 opacity-50" />}
                          </motion.div>
                        ))}
                      </div>
                    </div>

                    {/* Right panel: seat grid */}
                    <div className="flex-1 flex flex-col overflow-hidden">
                      <div className="px-4 pt-3 pb-1 flex items-center justify-between shrink-0">
                        <div>
                          <p className="text-[9px] uppercase font-black text-[#141414] tracking-widest">Seats</p>
                          <p className="text-[10px] font-bold text-[#141414] mt-0.5">
                            {allFilled
                              ? humanCount === 0 ? 'Spectator mode — no human placed' : 'Ready to start'
                              : `${setupSeats.filter(Boolean).length} / ${playerCount} filled`}
                          </p>
                        </div>
                        <button
                          onClick={handleRandomize}
                          className="text-[9px] uppercase font-black border-2 border-[#141414] text-[#141414] px-3 py-1.5 hover:bg-[#141414] hover:text-[#E4E3E0] transition-colors rounded"
                        >
                          {setupSeats.some(Boolean) ? 'Reshuffle' : 'Random'}
                        </button>
                      </div>

                      <div className="flex-1 overflow-y-auto p-3">
                        {numTeams > 1 ? (
                          /* Team mode: grouped by team */
                          <div className="space-y-3">
                            {Array.from({ length: numTeams }, (_, teamIdx) => {
                              const teamSize = Math.floor(playerCount / numTeams);
                              const tc = TEAM_COLOR_CLASSES[teamIdx % TEAM_COLOR_CLASSES.length];
                              return (
                                <div key={teamIdx} className={`border-2 ${tc.border} rounded-lg p-2.5`}>
                                  <p className={`text-[9px] uppercase font-black mb-2 ${tc.text}`}>Team {teamIdx + 1}</p>
                                  <div className="grid grid-cols-2 gap-1.5">
                                    {Array.from({ length: teamSize }, (_, j) => {
                                      const seatIdx = teamIdx * teamSize + j;
                                      const isHighlit = isDraggingSetup && !setupSeats[seatIdx];
                                      return (
                                        <div
                                          key={seatIdx}
                                          ref={el => { seatSlotRefs.current[seatIdx] = el; }}
                                          className={`min-h-[52px] border rounded flex items-center p-1 transition-colors
                                            ${isHighlit ? `${tc.border} ${tc.bg} border-2` : !setupSeats[seatIdx] ? 'border-black/50 bg-gray-50' : 'border-[#141414]/20 bg-white/60'}
                                            ${!setupSeats[seatIdx] ? 'border-dashed' : ''}`}
                                        >
                                          {setupSeats[seatIdx]
                                            ? <SeatCard seatIdx={seatIdx} />
                                            : <span className="text-[10px] uppercase font-bold text-[#141414] w-full text-center">Empty</span>}
                                        </div>
                                      );
                                    })}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        ) : (
                          /* FFA: simple grid */
                          <div className={`grid gap-2 ${playerCount <= 4 ? 'grid-cols-2' : playerCount <= 6 ? 'grid-cols-3' : 'grid-cols-4'}`}>
                            {Array.from({ length: playerCount }, (_, seatIdx) => {
                              const isHighlit = isDraggingSetup && !setupSeats[seatIdx];
                              return (
                                <div
                                  key={seatIdx}
                                  ref={el => { seatSlotRefs.current[seatIdx] = el; }}
                                  className={`min-h-[56px] border rounded flex items-center p-1.5 transition-colors
                                    ${isHighlit ? 'border-[#141414] bg-amber-50 border-2' : !setupSeats[seatIdx] ? 'border-black/50 bg-gray-50' : 'border-[#141414]/20 bg-white/60'}
                                    ${!setupSeats[seatIdx] ? 'border-dashed' : ''}`}
                                >
                                  {setupSeats[seatIdx]
                                    ? <SeatCard seatIdx={seatIdx} />
                                    : <span className="text-[10px] uppercase font-bold text-[#141414] w-full text-center">Empty</span>}
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </div>

                      {/* Poem */}
                      <div className="px-4 pb-2 pt-3 border-t border-black/20 shrink-0">
                        <p className="text-base italic text-[#141414] leading-relaxed">
                          Flip the stack — the suit is led<br />
                          Play a card or draw instead<br />
                          Follow suit, if suit you hold<br />
                          If you can't, a diamond's bold<br />
                          Match the rank — the suit will change<br />
                          Turn the table, rearrange
                        </p>
                      </div>

                      {/* Footer: back + start */}
                      <div className="border-t border-[#141414] p-3 flex gap-2 shrink-0">
                        <button
                          onClick={() => setSetupStep(1)}
                          className="py-2 px-4 border-2 border-[#141414] text-[#141414] text-[10px] font-black uppercase hover:bg-zinc-100 transition-colors rounded"
                        >
                          ← Back
                        </button>
                        <button
                          disabled={!canStart}
                          onClick={handleStart}
                          className="flex-1 py-2 bg-[#141414] text-[#E4E3E0] font-bold uppercase tracking-widest hover:bg-zinc-800 transition-colors disabled:opacity-40 rounded"
                        >
                          {humanCount === 0 ? 'Watch (Spectator)' : 'Start Game'}
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          );
        })()}

        {phase === 'ROUND_OVER' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-[#141414] z-50 flex items-center justify-center p-4"
          >
            <div className="w-full max-w-3xl flex gap-6 h-[80vh]">
              {/* Left — Scores + quote */}
              <div className="w-72 shrink-0 border-2 border-[#E4E3E0] bg-[#141414] text-[#E4E3E0] p-8 flex flex-col justify-center">
                <h2 className="text-3xl font-serif italic font-bold mb-1 uppercase text-center">Round Complete</h2>
                <p className="text-[10px] uppercase font-black opacity-40 tracking-widest text-center mb-6">
                  {roundsPlayedRef.current} of {numRounds} rounds
                </p>
                <div className="space-y-3 mb-6">
                  {numTeams > 1 ? (
                    new Array(numTeams).fill(0).map((_, teamIdx) => (
                      <div key={teamIdx} className="flex justify-between items-center border-b border-[#E4E3E0]/20 pb-2">
                        <span className="text-xs font-bold uppercase opacity-70">Team {teamIdx + 1}</span>
                        <span className="font-mono text-xl">{gameScores[teamIdx] || 0}</span>
                      </div>
                    ))
                  ) : (
                    players.map((p, i) => (
                      <div key={p.id} className="flex justify-between items-center border-b border-[#E4E3E0]/20 pb-2">
                        <span className="text-xs font-bold uppercase opacity-70">{p.name}</span>
                        <div className="flex items-center gap-3">
                          <span className="text-[10px] opacity-40">+{p.captured.length - p.hand.length}</span>
                          <span className="font-mono text-xl">{gameScores[i] || 0}</span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
                {roundWinQuote && (
                  <p className="text-amber-400 italic font-serif text-base text-center mb-6 leading-snug">
                    {roundWinQuote.name ? <><span className="not-italic text-[10px] uppercase font-black opacity-50 block mb-1">{roundWinQuote.name}</span></> : null}
                    "{roundWinQuote.quote}"
                  </p>
                )}
                <button
                  onClick={() => startRound()}
                  className="w-full py-4 bg-[#E4E3E0] text-[#141414] font-bold uppercase tracking-widest hover:bg-white transition-colors flex items-center justify-center gap-2 mt-auto"
                >
                  <RotateCcw size={18} /> Start Next Round
                </button>
              </div>

              {/* Right — Game log */}
              <div className="flex-1 flex flex-col border-2 border-[#E4E3E0]/20 overflow-hidden">
                <div className="px-4 py-2 bg-[#E4E3E0]/10 shrink-0">
                  <h2 className="text-[10px] uppercase font-black flex items-center gap-2 italic text-[#E4E3E0]/60">
                    <Layers size={12} className="text-blue-400" /> Game Log
                  </h2>
                </div>
                <div className="flex-1 overflow-y-auto p-4 font-mono text-sm space-y-2 custom-scrollbar">
                  {logs.map((log) => (
                    <div
                      key={log.id}
                      className={`pb-1 border-b border-white/5 last:border-0 leading-tight
                        ${log.type === 'PLAYER' ? 'text-blue-400 font-bold' :
                          log.type === 'CPU' ? 'text-red-400 italic' :
                          'text-zinc-500'}
                        ${log.text.startsWith('---') ? 'bg-amber-900/30 p-2 rounded text-amber-300 font-black text-center my-2 text-xs' : ''}
                      `}
                    >
                      <span className="opacity-30 text-[9px] mr-2">[{new Date(log.timestamp).toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' })}]</span>
                      <span className="font-black mr-1 text-[10px]">
                        {log.type === 'PLAYER' ? 'YOU:' : log.type === 'CPU' ? 'CPU:' : 'SYS:'}
                      </span>
                      {log.text}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {phase === 'GAME_OVER' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-[#141414] z-50 flex items-center justify-center p-4"
          >
            <div className="w-full max-w-3xl flex gap-6 h-[80vh]">
              {/* Left — Victory / Tie panel */}
              <div className="w-72 shrink-0 border-2 border-[#E4E3E0] bg-[#141414] text-[#E4E3E0] p-8 text-center flex flex-col justify-center">
                {tieGameQuote ? (
                  <>
                    <div className="mx-auto mb-6 w-16 h-16 border-2 border-amber-400 rounded-full flex items-center justify-center">
                      <span className="text-amber-400 text-2xl font-black">=</span>
                    </div>
                    <h2 className="text-4xl font-serif italic font-bold mb-2 uppercase text-amber-400">Deadlock</h2>
                    <p className="text-sm font-mono mb-6 opacity-60">The scores are level. No one breaks away.</p>
                    <div className="border border-amber-400/40 rounded-lg p-4 mb-6">
                      <p className="text-[9px] uppercase font-black tracking-widest text-amber-400 mb-2">{tieGameQuote.speaker}</p>
                      <p className="text-amber-300 italic font-serif text-base leading-snug">"{tieGameQuote.quote}"</p>
                    </div>
                  </>
                ) : (
                  <>
                    <Trophy size={64} className="mx-auto mb-6 text-amber-500" />
                    <h2 className="text-4xl font-serif italic font-bold mb-2 uppercase">Victory</h2>
                    <p className="text-lg font-mono mb-4">
                      {winner !== null
                        ? (numTeams > 1
                            ? `Team ${winner + 1} wins after ${numRounds} rounds!`
                            : players[winner].isAI
                              ? `${players[winner].name} wins after ${numRounds} rounds!`
                              : `You win after ${numRounds} rounds!`)
                        : `Game over after ${numRounds} rounds!`}
                    </p>
                    {winGameQuote && (
                      <p className="text-amber-400 italic font-serif text-lg mb-6 leading-snug">"{winGameQuote}"</p>
                    )}
                  </>
                )}
                <div className="space-y-2 mb-8 opacity-70 text-sm">
                  {numTeams > 1 ? (
                    new Array(numTeams).fill(0).map((_, teamIdx) => (
                      <div key={teamIdx} className="flex justify-between">
                        <span>Team {teamIdx + 1}</span>
                        <span className="font-mono">{gameScores[teamIdx] || 0}</span>
                      </div>
                    ))
                  ) : (
                    players.map((p, i) => (
                      <div key={p.id} className="flex justify-between">
                        <span>{p.name}</span>
                        <span className="font-mono">{gameScores[i] || 0}</span>
                      </div>
                    ))
                  )}
                </div>
                <button
                  onClick={resetGame}
                  className="w-full py-4 bg-[#E4E3E0] text-[#141414] font-bold uppercase tracking-widest hover:bg-white transition-colors"
                >
                  New Game
                </button>
              </div>

              {/* Right — Game log */}
              <div className="flex-1 flex flex-col border-2 border-[#E4E3E0]/20 overflow-hidden">
                <div className="px-4 py-2 bg-[#E4E3E0]/10 shrink-0">
                  <h2 className="text-[10px] uppercase font-black flex items-center gap-2 italic text-[#E4E3E0]/60">
                    <Layers size={12} className="text-blue-400" /> Game Log
                  </h2>
                </div>
                <div className="flex-1 overflow-y-auto p-4 font-mono text-sm space-y-2 custom-scrollbar">
                  {logs.map((log) => (
                    <div
                      key={log.id}
                      className={`pb-1 border-b border-white/5 last:border-0 leading-tight
                        ${log.type === 'PLAYER' ? 'text-blue-400 font-bold' :
                          log.type === 'CPU' ? 'text-red-400 italic' :
                          'text-zinc-500'}
                        ${log.text.startsWith('---') ? 'bg-amber-900/30 p-2 rounded text-amber-300 font-black text-center my-2 text-xs' : ''}
                      `}
                    >
                      <span className="opacity-30 text-[9px] mr-2">[{new Date(log.timestamp).toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' })}]</span>
                      <span className="font-black mr-1 text-[10px]">
                        {log.type === 'PLAYER' ? 'YOU:' : log.type === 'CPU' ? 'CPU:' : 'SYS:'}
                      </span>
                      {log.text}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
