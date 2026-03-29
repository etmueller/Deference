/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
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
  text: string;
  type: 'PLAYER' | 'CPU' | 'SYSTEM';
  timestamp: number;
}

type MessageType = 'info' | 'warning' | 'error' | 'success';

const SUITS = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES];
const RANKS = [
  Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN, 
  Rank.EIGHT, Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE
];

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

// Target score based on competing units (teams in team mode, players in FFA).
const calcTargetScore = (numPlayers: number, numTeams: number): number => {
  const units = numTeams > 1 ? numTeams : numPlayers;
  return Math.ceil(100 / units) + 2;
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

export default function App() {
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
  const [targetScore, setTargetScore] = useState(36);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const [votes, setVotes] = useState<{ [playerId: number]: 'KEEP' | 'END' }>({});
  const [lastCapture, setLastCapture] = useState<{ playerName: string, count: number } | null>(null);

  const addLog = useCallback((msg: string, type: 'PLAYER' | 'CPU' | 'SYSTEM' = 'SYSTEM') => {
    setLogs(prev => [{ text: msg, type, timestamp: Date.now() }, ...prev].slice(0, 100));
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
  const isTied = liveScores.filter(s => s === maxScore).length > 1 && maxScore > 0;
  
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

  const startRound = useCallback(() => {
    const newDeck = shuffle(createDeck());
    
    // Initialize players based on playerCount
    const newPlayers: Player[] = [];
    for (let i = 0; i < playerCount; i++) {
      newPlayers.push({
        id: i,
        name: i === 0 ? 'Player 1' : `CPU ${i}`,
        hand: newDeck.splice(0, 4),
        captured: [],
        isAI: i !== 0,
        hasActed: false
      });
    }

    setDeck(newDeck);
    setPile([]);
    setSide([]);
    setPlayers(newPlayers);
    setVotes({});
    
    // Only reset scores if it's the very first round
    if (!gameStarted) {
      if (numTeams > 1) {
        setGameScores(new Array(numTeams).fill(0));
      } else {
        setGameScores(new Array(playerCount).fill(0));
      }
    }
    
    const nextLeader = gameStarted ? (roundLeaderIndex + 1) % playerCount : 0;
    setRoundLeaderIndex(nextLeader);
    setTurnLeaderIndex(nextLeader);
    setCurrentPlayerIndex(nextLeader);
    
    setPhase('START');
    setDeferred(false);
    setLeadSuit(null);
    setTurnActionCount(0);
    setLastChallengerId(null);
    setLastCapture(null);
    setGameStarted(true);
    setLogs([]);
    addLog("--- NEW ROUND STARTED ---", 'SYSTEM');
    showMessage("Round started. Your turn to flip a card.");
  }, [playerCount, gameStarted, roundLeaderIndex, numTeams, addLog, showMessage]);

  const startTurn = useCallback(() => {
    if (deck.length === 0) {
      setPhase('VOTING');
      setVotes({});
      addLog("The Stack is empty! Players must vote: Reshuffle or End Round?", 'SYSTEM');
      showMessage("Stack empty — time to vote!", 'warning');
      return;
    }

    const flipped = deck[0];
    const remainingDeck = deck.slice(1);
    
    setDeck(remainingDeck);
    setPile(prev => [...prev, flipped]);
    setLastChallengerId(null);
    setPlayers(prev => prev.map(p => ({ ...p, hasActed: false })));
    
    if (flipped.isJoker) {
      // Joker flipped from Stack: flipper immediately wins all cards in Pile and Side
      const allCards = [...pile, ...side, flipped];
      addLog(`${currentPlayer?.name} flipped a JOKER from the Stack — wins all ${allCards.length} card${allCards.length !== 1 ? 's' : ''} instantly!`, currentPlayer?.isAI ? 'CPU' : 'PLAYER');
      showMessage(`${currentPlayer?.name} flipped a Joker — wins the pile instantly!`, 'success');
      setPlayers(prev => prev.map((p, idx) =>
        idx === currentPlayerIndex ? { ...p, captured: [...p.captured, ...allCards] } : p
      ));
      setPile([]);
      setSide([]);
      setPhase('START');
      // The next player clockwise flips to start the new turn
      setCurrentPlayerIndex((currentPlayerIndex + 1) % players.length);
    } else {
      setLeadSuit(flipped.suit);
      setDeferred(false);
      setPhase('ACTION');
      setTurnActionCount(0);
      addLog(`${currentPlayer?.name} flipped ${getRankLabel(flipped.rank)}${getSuitIcon(flipped.suit)}`, currentPlayer?.isAI ? 'CPU' : 'PLAYER');
      showMessage(`Lead suit is ${getSuitIcon(flipped.suit)}. Choose an action.`);
    }
  }, [deck, pile, side, currentPlayerIndex, players, currentPlayer, addLog, showMessage]);

  const handleVote = (playerId: number, choice: 'KEEP' | 'END') => {
    // Capture fresh values at call time to avoid stale closures in the setTimeout
    const currentPlayers = players;
    const currentGameScores = gameScores;

    setVotes(prev => {
      const newVotes = { ...prev, [playerId]: choice };
      const voterName = currentPlayers[playerId].name;
      addLog(`${voterName} voted to ${choice === 'KEEP' ? 'Reshuffle' : 'End Round'}.`, currentPlayers[playerId].isAI ? 'CPU' : 'PLAYER');

      // Check if all players have voted
      if (Object.keys(newVotes).length === currentPlayers.length) {
        const keepVotes = Object.values(newVotes).filter(v => v === 'KEEP').length;
        const endVotes = Object.values(newVotes).filter(v => v === 'END').length;

        let outcome: 'KEEP' | 'END';
        let logMsg = "";

        if (keepVotes > endVotes) {
          outcome = 'KEEP';
          logMsg = `Majority voted to KEEP PLAYING (${keepVotes} vs ${endVotes}).`;
        } else if (endVotes > keepVotes) {
          outcome = 'END';
          logMsg = `Majority voted to END ROUND (${endVotes} vs ${keepVotes}).`;
        } else {
          // Tie: Coin flip
          const flip = Math.random() > 0.5 ? 'KEEP' : 'END';
          outcome = flip;
          logMsg = `It's a TIE (${keepVotes} vs ${endVotes})! System coin flip decided: ${flip === 'KEEP' ? 'KEEP PLAYING' : 'END ROUND'}.`;
        }

        addLog(`--- VOTE OUTCOME: ${outcome === 'KEEP' ? 'RESHUFFLE' : 'END ROUND'} ---`, 'SYSTEM');
        addLog(logMsg, 'SYSTEM');

        setTimeout(() => {
          if (outcome === 'KEEP') {
            // Reshuffle all captured cards
            const allCaptured = currentPlayers.flatMap(p => p.captured);
            if (allCaptured.length === 0) {
              addLog("No captured cards to reshuffle! Ending round instead.", 'SYSTEM');
              const isGameOver = calculateRoundScores(currentPlayers, currentGameScores);
              if (!isGameOver) setPhase('ROUND_OVER');
            } else {
              const newDeck = shuffle(allCaptured);
              setDeck(newDeck);
              setPlayers(prev => prev.map(p => ({ ...p, captured: [] })));
              setPhase('START');
              showMessage("Deck reshuffled. Continue playing!");
            }
          } else {
            const isGameOver = calculateRoundScores(currentPlayers, currentGameScores);
            if (!isGameOver) setPhase('ROUND_OVER');
          }
        }, 2000);
      }
      return newVotes;
    });
  };

  const calculateRoundScores = useCallback((currentPlayers: Player[], currentGameScores: number[]) => {
    const roundScores = currentPlayers.map(p => p.captured.length - p.hand.length);
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
    } else {
      newGameScores = currentGameScores.map((s, i) => s + (roundScores[i] || 0));
      addLog("--- ROUND OVER ---");
      currentPlayers.forEach((p, i) => addLog(`${p.name}: ${roundScores[i]} pts (Total: ${newGameScores[i]})`));
    }

    setGameScores(newGameScores);

    if (newGameScores.some(s => s >= targetScore)) {
      setPhase('GAME_OVER');
      const maxScore = Math.max(...newGameScores);
      const winnerIdx = newGameScores.indexOf(maxScore);
      setWinner(winnerIdx);

      if (numTeams > 1) {
        addLog(`GAME OVER! Team ${winnerIdx + 1} wins!`);
      } else {
        addLog(`GAME OVER! ${currentPlayers[winnerIdx].name} wins!`);
      }
      return true; // Game is over
    }
    return false; // Game continues
  }, [numTeams, targetScore, addLog]);

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
      addLog(`${currentPlayer?.name} passed.`);
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
        addLog(`${currentPlayer?.name} played Joker!`, currentPlayer?.isAI ? 'CPU' : 'PLAYER');
        showMessage(`${currentPlayer?.name} played a Joker — wins the pile!`, 'success');
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
        addLog(`${currentPlayer?.name} matched rank ${getRankLabel(card.rank)} and Suit Switched to ${getSuitIcon(card.suit)}`, currentPlayer?.isAI ? 'CPU' : 'PLAYER');
        showMessage(`${currentPlayer?.name} Suit Switched to ${getSuitIcon(card.suit)}!`);
      } else {
        // Lead suit play or Diamond Defer
        if (isDiamond && !isLeadSuit && !deferred) {
          setDeferred(true);
          addLog(`${currentPlayer?.name} deferred the pile.`, currentPlayer?.isAI ? 'CPU' : 'PLAYER');
          showMessage(`${currentPlayer?.name} deferred the pile!`, 'warning');
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
      setCurrentPlayerIndex((currentPlayerIndex + 1) % players.length);
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
      const nextTurnLeader = (turnLeaderIndex + 1) % players.length;
      setTurnLeaderIndex(nextTurnLeader);
      setCurrentPlayerIndex(nextTurnLeader);
    }

    // Check for Round End at the end of the TURN
    // Condition: Any player has no cards left OR (optionally) pile is empty (meaning it was just won)
    // We prioritize the "no cards left" as the primary round end trigger.
    const someoneOut = updatedPlayers.some(p => p.hand.length === 0);
    
    if (someoneOut) {
      const isGameOver = calculateRoundScores(updatedPlayers, gameScores);
      if (!isGameOver) {
        setPhase('ROUND_OVER');
      }
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
    setLastChallengerId(null);
    setGameStarted(false);
    setRoundLeaderIndex(0);
    setTurnLeaderIndex(0);
    setLogs([]);
    setVotes({});
    setLastCapture(null);
  }, []);

  useEffect(() => {
    if (phase === 'VOTING') {
      const timer = setTimeout(() => {
        players.forEach(p => {
          if (p.isAI && votes[p.id] === undefined) {
            const choice = Math.random() > 0.5 ? 'KEEP' : 'END';
            handleVote(p.id, choice);
          }
        });
      }, 1000);
      return () => clearTimeout(timer);
    }

    if (currentPlayer?.isAI) {
      if (phase === 'START') {
        const timer = setTimeout(() => {
          startTurn();
        }, 1000);
        return () => clearTimeout(timer);
      }

      if (phase === 'ACTION') {
        const timer = setTimeout(() => {
          // AI Logic: Respect strict suit rules
          const hasLeadSuit = currentPlayer?.hand.some(c => c.suit === leadSuit && !c.isJoker);
          const pileTop = pile.length > 0 ? pile[pile.length - 1] : null;
          
          let playable: Card[] = [];
          if (hasLeadSuit) {
            // Must play lead suit, Joker, or Rank Match
            playable = currentPlayer?.hand.filter(c => 
              c.suit === leadSuit || 
              c.isJoker || 
              (pileTop && c.rank === pileTop.rank)
            ) || [];
          } else {
            // Can play Joker, Rank Match, or Diamond (Defer)
            playable = currentPlayer?.hand.filter(c => 
              c.isJoker || 
              (pileTop && c.rank === pileTop.rank) ||
              (c.suit === Suit.DIAMONDS)
            ) || [];
          }

          if (playable.length > 0) {
            // Prefer Joker if it wins, or Suit Switch if no lead suit
            const joker = playable.find(c => c.isJoker);
            if (joker) {
              handleAction('PLAY', joker);
              return;
            }

            const switchCard = !hasLeadSuit ? playable.find(c => pile.length > 0 && c.rank === pile[pile.length - 1].rank) : null;
            if (switchCard) {
              handleAction('PLAY', switchCard);
              return;
            }

            // Play highest playable
            const best = playable.sort((a, b) => b.rank - a.rank)[0];
            handleAction('PLAY', best);
          } else if (deck.length > 0) {
            handleAction('DRAW');
          } else {
            handleAction('PASS');
          }
        }, 1500);
        return () => clearTimeout(timer);
      }
    }
  }, [phase, currentPlayerIndex, leadSuit, pile, deck.length, startTurn, currentPlayer?.isAI, votes, players]);

  // --- Render Helpers ---

  const sideTop = side.length > 0 ? side[side.length - 1] : null;
  const pileTop = pile.length > 0 ? pile[pile.length - 1] : null;

  // Adaptive scoreboard sizing based on player count
  const sbConfig = playerCount <= 2
    ? { scoreSz: 'text-3xl',     nameSz: 'text-sm',     pad: 'px-3 py-2.5',   statSz: 'text-[13px]', useGrid: false }
    : playerCount <= 4
    ? { scoreSz: 'text-[28px]',  nameSz: 'text-sm',     pad: 'px-2 py-2',     statSz: 'text-[13px]', useGrid: false }
    : playerCount <= 5
    ? { scoreSz: 'text-2xl',     nameSz: 'text-xs',     pad: 'px-2 py-1.5',   statSz: 'text-[11px]', useGrid: false }
    : { scoreSz: 'text-xl',      nameSz: 'text-[10px]', pad: 'px-1.5 py-1',   statSz: 'text-[10px]', useGrid: true  };

  return (
    <div className="h-screen bg-[#E4E3E0] text-[#141414] font-sans selection:bg-[#141414] selection:text-[#E4E3E0] flex flex-col overflow-hidden">
      {/* Header - Minimal */}
      <header className="border-b border-[#141414] px-4 py-2 flex justify-between items-center bg-white/30 shrink-0">
        <div className="flex items-baseline gap-2">
          <h1 className="text-xl font-bold tracking-tighter uppercase italic font-serif">DEFERENCE</h1>
          <p className="text-[10px] uppercase tracking-widest opacity-60 font-bold">Diamonds are for the clever.</p>
        </div>
        <div className="flex gap-4 items-center">
          <div className="text-right flex items-center gap-2">
            <p className="text-[10px] uppercase opacity-60 font-bold">Target</p>
            <p className="font-mono font-bold text-lg">{targetScore}</p>
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
              {currentPlayer?.name}'S TURN
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
            </div>
          </div>

        </div>

        {/* BOTTOM-LEFT: Your Hand + Buttons */}
        <div className="border-r border-[#141414] flex flex-col p-2 bg-white/40 row-start-2 col-start-1">
            <div className="flex items-center gap-2 mb-1">
              <ArrowDownCircle size={14} className="text-amber-600" />
              <span className="text-[10px] uppercase font-black">Your Hand</span>
            </div>

            <div className="flex justify-center items-center gap-1 overflow-x-hidden py-1">
              <AnimatePresence>
                {players.length > 0 && players[0].hand.map((card, idx) => (
                  <CardView 
                    key={card.id} 
                    card={card} 
                    size={players[0].hand.length > 8 ? "xs" : players[0].hand.length > 5 ? "sm" : "md"}
                    onClick={() => handleAction('PLAY', card)}
                    disabled={phase !== 'ACTION' || currentPlayer?.isAI}
                  />
                ))}
              </AnimatePresence>
              {players.length > 0 && players[0].hand.length === 0 && phase !== 'VOTING' && (
                <div className="flex items-center gap-2 opacity-20 italic text-xs">
                  <AlertCircle size={14} /> No cards
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="flex justify-center gap-2 mt-2">
              <button 
                disabled={phase !== 'ACTION' || currentPlayer?.isAI}
                onClick={() => handleAction('DRAW')}
                className="flex-1 max-w-[100px] py-1.5 border border-[#141414] text-[10px] font-black uppercase hover:bg-[#141414] hover:text-[#E4E3E0] disabled:opacity-30 transition-all rounded shadow-[2px_2px_0px_0px_#141414] active:translate-x-[1px] active:translate-y-[1px] active:shadow-none bg-white"
              >
                Draw
              </button>
              <button 
                disabled={phase !== 'ACTION' || currentPlayer?.isAI}
                onClick={() => handleAction('PASS')}
                className="flex-1 max-w-[100px] py-1.5 border border-[#141414] text-[10px] font-black uppercase hover:bg-[#141414] hover:text-[#E4E3E0] disabled:opacity-30 transition-all rounded shadow-[2px_2px_0px_0px_#141414] active:translate-x-[1px] active:translate-y-[1px] active:shadow-none bg-white"
              >
                Pass
              </button>
            </div>
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
                            {isLeader && (
                              <span className="flex items-center gap-0.5 bg-amber-400 text-[#141414] px-1 py-0.5 rounded text-[7px] font-black not-italic shrink-0">
                                <Trophy size={7} /> {isTied ? 'TIE' : 'LEAD'}
                              </span>
                            )}
                          </div>
                          <span className={`font-mono font-black shrink-0 ml-1 ${sbConfig.scoreSz} ${isLeader ? 'text-amber-600' : 'text-[#141414]'}`}>
                            {score}
                          </span>
                        </div>
                        {/* Individual player sub-rows */}
                        <div className="divide-y divide-[#141414]/5 border-t border-[#141414]/10">
                          {teamPlayers.map(p => (
                            <div
                              key={p.id}
                              className={`flex justify-between items-center px-2 py-0.5 ${sbConfig.statSz} transition-colors
                                ${currentPlayerIndex === p.id ? 'bg-blue-500 text-white' : 'opacity-60'}`}
                            >
                              <div className="flex items-center gap-0.5">
                                {p.isAI ? <Cpu size={7} /> : <User size={7} />}
                                <span className="font-bold truncate max-w-[40px]">{p.name}</span>
                                {p.hasActed && <span className={`font-black text-[11px] ${currentPlayerIndex === p.id ? 'text-white' : 'text-green-500'}`}>✓</span>}
                              </div>
                              <div className="flex gap-1.5 font-mono">
                                <span>H:{p.hand.length}</span>
                                <span>C:{p.captured.length}</span>
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
                          ${p.hasActed && !isCurrent ? 'opacity-50' : ''}
                        `}
                      >
                        <div className="flex flex-col min-w-0">
                          <div className="flex items-center gap-1">
                            <span className={`${sbConfig.nameSz} font-black uppercase italic flex items-center gap-0.5 truncate ${isLeader ? 'text-amber-700' : ''}`}>
                              {p.isAI ? <Cpu size={8} /> : <User size={8} />} {p.name}
                            </span>
                            {p.hasActed && <span className="text-green-500 font-black text-[11px] shrink-0">✓</span>}
                            {isLeader && (
                              <span className="flex items-center gap-0.5 bg-amber-400 text-[#141414] px-1 py-0.5 rounded text-[7px] font-black not-italic shrink-0">
                                <Trophy size={7} /> {isTied ? 'TIE' : 'LEAD'}
                              </span>
                            )}
                          </div>
                          <div className={`flex gap-1.5 mt-0.5 font-bold text-zinc-400 ${sbConfig.statSz}`}>
                            <span>H:{p.hand.length}</span>
                            <span>C:{p.captured.length}</span>
                          </div>
                        </div>
                        <span className={`font-mono font-black shrink-0 ml-1 ${sbConfig.scoreSz} ${isLeader ? 'text-amber-600' : 'text-[#141414]'}`}>
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
                {logs.map((log, i) => (
                  <motion.div 
                    key={log.timestamp + i}
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
        {phase === 'VOTING' && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-[#141414]/80 backdrop-blur-md z-[60] flex items-center justify-center p-4"
          >
            <div className="max-w-md w-full border-4 border-[#141414] bg-[#E4E3E0] p-8 text-center shadow-[10px_10px_0px_0px_#141414] rounded-3xl">
              <Layers size={48} className="mx-auto mb-4 text-[#141414]" />
              <h2 className="text-2xl font-serif italic font-black mb-2 uppercase tracking-tighter">Stack Depleted</h2>
              <p className="text-sm mb-8 font-bold">The round isn't over yet! Should we reshuffle the captured cards and keep playing, or end the round now?</p>
              
              <div className="grid grid-cols-2 gap-4">
                <button 
                  disabled={votes[0] !== undefined}
                  onClick={() => handleVote(0, 'KEEP')}
                  className={`py-4 rounded-xl border-2 border-[#141414] font-black uppercase text-sm shadow-[4px_4px_0px_0px_#141414] active:translate-x-[2px] active:translate-y-[2px] active:shadow-none transition-all
                    ${votes[0] === 'KEEP' ? 'bg-[#141414] text-[#E4E3E0]' : 'bg-white hover:bg-amber-50'}`}
                >
                  Reshuffle
                </button>
                <button 
                  disabled={votes[0] !== undefined}
                  onClick={() => handleVote(0, 'END')}
                  className={`py-4 rounded-xl border-2 border-[#141414] font-black uppercase text-sm shadow-[4px_4px_0px_0px_#141414] active:translate-x-[2px] active:translate-y-[2px] active:shadow-none transition-all
                    ${votes[0] === 'END' ? 'bg-[#141414] text-[#E4E3E0]' : 'bg-white hover:bg-red-50'}`}
                >
                  End Round
                </button>
              </div>

              <div className="mt-8 space-y-2">
                <p className="text-[10px] uppercase font-black opacity-40 mb-2">Voting Status</p>
                <div className="flex flex-wrap justify-center gap-2">
                  {players.map(p => (
                    <div key={p.id} className={`px-3 py-1 rounded-full border border-[#141414] text-[9px] font-bold flex items-center gap-1.5
                      ${votes[p.id] ? 'bg-green-500 text-white border-green-600' : 'bg-white/50 opacity-50'}`}>
                      {p.isAI ? <Cpu size={10} /> : <User size={10} />}
                      {p.name}: {votes[p.id] ? 'VOTED' : '...'}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
        {!gameStarted && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-[#E4E3E0]/90 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          >
            <div className="max-w-md w-full border-2 border-[#141414] bg-white p-8 text-center shadow-[10px_10px_0px_0px_#141414]">
              <h2 className="text-4xl font-serif italic font-bold mb-4 tracking-tighter">DEFERENCE</h2>
              <p className="text-sm mb-8 opacity-70 italic">Diamonds are for the clever.</p>
              
              <div className="space-y-6 mb-8 text-left">
                {/* Player count */}
                <div>
                  <label className="text-[10px] uppercase font-bold opacity-50 block mb-2">Number of Players</label>
                  <div className="flex gap-1.5">
                    {[2, 3, 4, 5, 6, 7, 8].map(n => (
                      <button
                        key={n}
                        onClick={() => {
                          setPlayerCount(n);
                          setNumTeams(1);
                          setTargetScore(calcTargetScore(n, 1));
                        }}
                        className={`flex-1 py-2 border border-[#141414] text-xs font-mono ${playerCount === n ? 'bg-[#141414] text-[#E4E3E0]' : 'hover:bg-zinc-100'}`}
                      >
                        {n}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Game mode — only for even counts with team options */}
                {TEAM_MODES[playerCount] && (
                  <div>
                    <label className="text-[10px] uppercase font-bold opacity-50 block mb-2">Game Mode</label>
                    <div className="flex gap-1.5 flex-wrap">
                      {TEAM_MODES[playerCount].map(mode => (
                        <button
                          key={mode.numTeams}
                          onClick={() => {
                            setNumTeams(mode.numTeams);
                            setTargetScore(calcTargetScore(playerCount, mode.numTeams));
                          }}
                          className={`flex-1 py-2 border border-[#141414] text-xs font-mono ${numTeams === mode.numTeams ? 'bg-[#141414] text-[#E4E3E0]' : 'hover:bg-zinc-100'}`}
                        >
                          {mode.label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Target score */}
                <div>
                  <label className="text-[10px] uppercase font-bold opacity-50 block mb-2">
                    Target Score
                    <span className="ml-2 normal-case opacity-70">
                      ⌈100 ÷ {numTeams > 1 ? numTeams : playerCount}⌉ + 2
                    </span>
                  </label>
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      min="10"
                      max="100"
                      step="1"
                      value={targetScore}
                      onChange={(e) => setTargetScore(parseInt(e.target.value))}
                      className="flex-1 accent-[#141414]"
                    />
                    <span className="font-mono font-bold text-xl w-12 text-center">{targetScore}</span>
                  </div>
                </div>
              </div>

              <button 
                onClick={startRound}
                className="w-full py-4 bg-[#141414] text-[#E4E3E0] font-bold uppercase tracking-widest hover:bg-zinc-800 transition-colors"
              >
                Initialize Operation
              </button>
            </div>
          </motion.div>
        )}

        {phase === 'ROUND_OVER' && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-[#E4E3E0]/90 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          >
            <div className="max-w-md w-full border-2 border-[#141414] bg-white p-8 text-center shadow-[10px_10px_0px_0px_#141414]">
              <h2 className="text-2xl font-serif italic font-bold mb-4 uppercase">Round Complete</h2>
              <p className="text-xs opacity-50 mb-6 italic">Review the game log and scores before proceeding.</p>
              <div className="space-y-4 mb-8">
                {numTeams > 1 ? (
                  <>
                    {new Array(numTeams).fill(0).map((_, teamIdx) => (
                      <div key={teamIdx} className="flex justify-between items-center border-b border-[#141414] pb-2">
                        <span className="text-xs font-bold uppercase">Team {teamIdx + 1}</span>
                        <div className="flex items-center gap-4">
                          <span className="font-mono text-xl">{gameScores[teamIdx] || 0}</span>
                        </div>
                      </div>
                    ))}
                  </>
                ) : (
                  players.map((p, i) => (
                    <div key={p.id} className="flex justify-between items-center border-b border-[#141414] pb-2">
                      <span className="text-xs font-bold uppercase">{p.name}</span>
                      <div className="flex items-center gap-4">
                        <span className="text-[10px] opacity-50">Round: {p.captured.length - p.hand.length}</span>
                        <span className="font-mono text-xl">{gameScores[i] || 0}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
              <button 
                onClick={startRound}
                className="w-full py-4 bg-[#141414] text-[#E4E3E0] font-bold uppercase tracking-widest hover:bg-zinc-800 transition-colors flex items-center justify-center gap-2"
              >
                <RotateCcw size={18} /> Start Next Round
              </button>
            </div>
          </motion.div>
        )}

        {phase === 'GAME_OVER' && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-[#141414] z-50 flex items-center justify-center p-4"
          >
            <div className="max-w-md w-full border-2 border-[#E4E3E0] bg-[#141414] text-[#E4E3E0] p-8 text-center">
              <Trophy size={64} className="mx-auto mb-6 text-amber-500" />
              <h2 className="text-4xl font-serif italic font-bold mb-2 uppercase">Victory</h2>
              <p className="text-xl font-mono mb-8">
                {winner !== null ? (numTeams > 1 ? `Team ${winner + 1}` : players[winner].name) : 'Unknown'} has reached the target score!
              </p>
              <div className="space-y-2 mb-8 opacity-70">
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
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
