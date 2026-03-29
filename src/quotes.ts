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
