export type QuoteEvent = 'WIN_TURN' | 'WIN_ROUND' | 'WIN_GAME';

const QUOTES: Record<string, Record<QuoteEvent, string[]>> = {
  You: {
    WIN_TURN: [
      "Mine.",
      "Too easy.",
      "Thank you very much.",
      "I'll take that.",
      "Don't mind if I do.",
    ],
    WIN_ROUND: [
      "Not bad for a human.",
      "Round goes to me.",
      "I could get used to this.",
      "Keep underestimating me.",
    ],
    WIN_GAME: [
      "And that's how it's done.",
      "You can all go home now.",
      "Next time, bring a strategy.",
    ],
  },
  Lucy: {
    WIN_TURN: [
      "Obviously.",
      "Did you even try?",
      "Better luck next pile.",
      "I saw that coming a mile away.",
      "Predictable.",
    ],
    WIN_ROUND: [
      "Round to Lucy. As expected.",
      "You're all so easy to read.",
      "Was there ever any doubt?",
      "Try harder next round.",
    ],
    WIN_GAME: [
      "Flawless. As always.",
      "I could have done this blindfolded.",
      "Don't feel bad. I'm just better.",
    ],
  },
  Zane: {
    WIN_TURN: [
      "Boom!",
      "Get rekt.",
      "Too slow!",
      "Zane takes it, obviously.",
      "Speed wins every time.",
    ],
    WIN_ROUND: [
      "Round over, Zane wins, let's go!",
      "The fast always beat the slow.",
      "Keep up, people!",
      "Another round, another win.",
    ],
    WIN_GAME: [
      "Nobody was keeping up anyway.",
      "That's what peak performance looks like.",
      "First place. No surprises there.",
    ],
  },
  Uncle: {
    WIN_TURN: [
      "Ah, the classics never fail.",
      "I've been playing this since before you were born.",
      "Every card has its moment.",
      "Experience counts for something.",
      "Patience, my friends.",
    ],
    WIN_ROUND: [
      "The old ways still work.",
      "Another round for the veteran.",
      "They said I was past it. Ha.",
      "Age and guile beat youth and speed.",
    ],
    WIN_GAME: [
      "They don't make opponents like they used to.",
      "Another game, another lesson taught.",
      "Retirement suits a winner.",
    ],
  },
  Barney: {
    WIN_TURN: [
      "Heh. Didn't see that coming, did ya?",
      "Lucky again!",
      "Barney's on a roll!",
      "You snooze, I win.",
      "Ha!",
    ],
    WIN_ROUND: [
      "I'm unstoppable today!",
      "Barney wins the round, baby!",
      "Luck? Skill? Who cares — it worked!",
      "Another round bites the dust.",
    ],
    WIN_GAME: [
      "I can't believe it — I won!",
      "Barney WINS! Write that down!",
      "Did NOT see that coming. Neither did you.",
    ],
  },
  Fabi: {
    WIN_TURN: [
      "Calculated.",
      "Just as planned.",
      "The numbers don't lie.",
      "I ran the probabilities.",
      "Optimal play.",
    ],
    WIN_ROUND: [
      "The variance favoured me this round.",
      "Expected value: positive.",
      "My model was correct.",
      "Efficiency wins rounds.",
    ],
    WIN_GAME: [
      "The optimal strategy pays off.",
      "Statistically, this was the most likely outcome.",
      "Game theory: verified.",
    ],
  },
  Jane: {
    WIN_TURN: [
      "Oh! I got it!",
      "Yay!",
      "That was fun!",
      "Can we go again?",
      "I didn't mess up this time!",
    ],
    WIN_ROUND: [
      "I actually won a round!",
      "This is so exciting!",
      "Nobody expected that. Especially me.",
      "Round for Jane!",
    ],
    WIN_GAME: [
      "Oh my goodness, I won the whole thing!",
      "I have no idea how that happened but I'll take it!",
      "Best day ever!",
    ],
  },
  Noema: {
    WIN_TURN: [
      "Quietly taken.",
      "No drama.",
      "It was always going to be mine.",
      "Still waters run deep.",
      "Patience rewarded.",
    ],
    WIN_ROUND: [
      "The round belongs to me.",
      "Steady wins.",
      "No need for celebration.",
      "As it should be.",
    ],
    WIN_GAME: [
      "It was never in doubt.",
      "The game was decided long ago.",
      "Composure. That's all it takes.",
    ],
  },
};

export const TIE_GAME_QUOTES: Record<string, string[]> = {
  You: [
    "You didn't win. You didn't lose. You held the line.",
    "A deadlock. Nobody takes the pile. Nobody takes the crown.",
    "You matched them move for move. No separation.",
    "No winner. Just tension unresolved.",
    "You stopped the game from choosing.",
    "Balance, for once. Nobody walks away ahead.",
    "You refused to break. They couldn't either.",
    "A draw. Not failure — containment.",
    "You met them exactly where they stood.",
    "No one proved more. No one proved less.",
  ],
  Jane: [
    "In chess, a tie is worth half a point.",
    "Longest chess game on record: 269 moves, 20+ hours... and it ended in a draw.",
    "A perfect stalemate — every move answered, every play matched.",
    "No edge found, no ground lost. That's as even as it gets.",
    "Skill met skill and refused to give way.",
    "A deadlock like this usually means both sides played it just right.",
    "Sometimes the best outcome is proving you're equals.",
    "We reach our denouement — Pay deference and carry on.",
    "And the Joker never came — A tie also wins the game.",
  ],
  Noema: [
    "A perfect mirror image — you two are exactly as good as each other.",
    "You pushed each other to the same finish line — impressive symmetry.",
    "You cancelled each other out in the most impressive way.",
    "Neither of you blinked — that's why the scoreboard couldn't either.",
    "Harmony on the scoreboard: a perfect balance of skill.",
  ],
  Fabi: [
    "A tie means the game was too good for anyone to lose.",
    "Why settle for one winner when you can have two?",
    "Friends to the end — not even the score could pull you apart.",
    "Call it a draw, call it respect — either way, well played.",
    "No winner, but plenty of quality on display.",
    "Double the winners, zero losers. Now that's a result.",
  ],
  Barney: [
    "They say a tie is like kissing your sister. How's it feel, Kingslayer?",
    "We didn't lose; we just ran out of time to win.",
    "A tie is just a win that hasn't made up its mind yet.",
    "Great minds play alike!",
  ],
  Zane: [
    "If we all tied, then we'd all be in first!",
    "Why pick a winner when chaos says everyone wins?",
    "You both broke the game so hard it couldn't decide.",
    "Hot off the press: Harvard Beats Yale 29-29.",
  ],
  Uncle: [
    "A famous 0-0 tie: Army vs. Notre Dame in 1946 — one of the greatest games ever played.",
    "Instant classic, just like when the Panthers tied the Bengals at 37 back in '14.",
    "Soccer is the greatest game in the world, because of the tie.",
    "Back in my day, a tie like this meant both sides earned it.",
  ],
  Lucy: [
    "If we all tied every match like you just did, we'd all be sitting in first place together.",
    "No losers, just two winners sharing the spotlight.",
    "Great minds play alike!",
    "That's what happens when two great players meet.",
    "Everyone walks away proud of that one.",
  ],
};

export function getTieGameQuote(players: { name: string; isAI: boolean }[]): { speaker: string; quote: string } {
  const speaker = players[Math.floor(Math.random() * players.length)];
  const key = speaker.isAI ? speaker.name : 'You';
  // Fall back to base name for duplicates (e.g. "Noema A" → "Noema")
  const list = TIE_GAME_QUOTES[key] ?? TIE_GAME_QUOTES[key.split(' ')[0]] ?? TIE_GAME_QUOTES['You'];
  return { speaker: speaker.isAI ? speaker.name : 'You', quote: pick(list) };
}

const TIE_QUOTES: string[] = [
  "A draw? Fitting.",
  "Nobody wins, nobody loses. How boring.",
  "Perfectly balanced, as all things should be.",
  "Too close to call.",
  "Share and share alike.",
  "Well, that was inconclusive.",
  "The tie stands. Rematch?",
];

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

export function getQuote(playerName: string, event: QuoteEvent): string {
  const playerQuotes = QUOTES[playerName];
  if (!playerQuotes) return '';
  return pick(playerQuotes[event]);
}

export function getTieQuote(): string {
  return pick(TIE_QUOTES);
}
