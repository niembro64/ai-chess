// Cartoon bot portraits for the lobby's model grid.
//
// Inline SVG bodies (no <svg> wrapper — BotIcon.vue supplies it with a
// 0 0 64 64 viewBox), same pattern as pieceIcons.ts. Each bot has TWO
// moods, because the grid crosses model × goal:
//
//   sage    — calm when asked to win (its nature), flustered when
//             asked to lose
//   jester  — gleeful when asked to lose (its nature), straining when
//             asked to win
//
// Palettes match the piece tints in models.ts: Sage green, Jester
// purple, Toy teal.

const SKIN = '#f4dcc0';
const SAGE_HAT = '#3f9e63';
const SAGE_HAT_LIT = '#57b97a';
const SAGE_HAT_DARK = '#2c7a4b';
const JEST_HAT = '#a855f7';
const JEST_HAT_LIT = '#c084fc';
const JEST_HAT_DARK = '#7c3aed';
const BELL = '#ffd76a';
const HAIR = '#eef2ee';
const HAIR_SHADE = '#dde3dd';
const INK = '#2b2438';

// --- Sage: pointed star-hat, long beard --------------------------------

// Hat sits high and the beard is deliberately NARROWER than the face
// circle — an early version hung a full-width beard from the eyes down
// and the sage rendered as a white blob with two dots.
const sageHat = `
  <path d="M32 2 C25 9 18 17 14 23 C22 19.5 42 19.5 50 23 C46 17 39 9 32 2 Z" fill="${SAGE_HAT}"/>
  <path d="M32 2 C25 9 18 17 14 23 C18 21.5 22 20.6 26 20.2 C27 13.5 29 7 32 2 Z" fill="${SAGE_HAT_LIT}"/>
  <circle cx="30.5" cy="13.5" r="1.7" fill="${BELL}"/>
  <circle cx="35.5" cy="19" r="1.1" fill="${BELL}"/>
  <rect x="12" y="22" width="40" height="5.5" rx="2.75" fill="${SAGE_HAT_DARK}"/>
`;

const sageBeard = `
  <path d="M23 42 C22 53.5 26 58.5 32 58.5 C38 58.5 42 53.5 41 42 C37 46 27 46 23 42 Z" fill="${HAIR}"/>
  <path d="M24.5 42.5 C27.5 45.5 36.5 45.5 39.5 42.5 C36.5 44 27.5 44 24.5 42.5 Z" fill="${HAIR_SHADE}"/>
`;

export const SAGE_CALM = `
  ${sageHat}
  <circle cx="32" cy="37" r="13" fill="${SKIN}"/>
  ${sageBeard}
  <path d="M24 31 C26.5 29 30 29.3 32 31" stroke="${HAIR}" stroke-width="2.1" fill="none" stroke-linecap="round"/>
  <path d="M32 31 C34 29.3 37.5 29 40 31" stroke="${HAIR}" stroke-width="2.1" fill="none" stroke-linecap="round"/>
  <path d="M25 37 C27 33.8 30 33.8 31.5 37" stroke="${INK}" stroke-width="1.9" fill="none" stroke-linecap="round"/>
  <path d="M32.5 37 C34 33.8 37 33.8 39 37" stroke="${INK}" stroke-width="1.9" fill="none" stroke-linecap="round"/>
`;

export const SAGE_FLUSTERED = `
  <g transform="rotate(-13 32 26)">${sageHat}</g>
  <circle cx="32" cy="37" r="13" fill="${SKIN}"/>
  ${sageBeard}
  <path d="M23.5 31.5 C26 28.8 29 28.2 31.5 29" stroke="${HAIR}" stroke-width="2.1" fill="none" stroke-linecap="round"/>
  <path d="M32.5 29 C35 28.2 38 28.8 40.5 31.5" stroke="${HAIR}" stroke-width="2.1" fill="none" stroke-linecap="round"/>
  <ellipse cx="27" cy="36.5" rx="3.8" ry="4.3" fill="#ffffff"/>
  <ellipse cx="37" cy="36.5" rx="3.8" ry="4.3" fill="#ffffff"/>
  <circle cx="28.6" cy="37.4" r="1.9" fill="${INK}"/>
  <circle cx="35.4" cy="37.4" r="1.9" fill="${INK}"/>
  <path d="M47 27 C49.6 31.4 50.8 34 50.8 35.8 C50.8 38.3 49 39.8 47 39.8 C45 39.8 43.2 38.3 43.2 35.8 C43.2 34 44.4 31.4 47 27 Z" fill="#8fd4ff"/>
`;

// --- Jester: three-point belled cap, big grin --------------------------

const jesterHat = `
  <path d="M32 16 C23 17 15 13 9 6 C6 14 10 22 19 24 Z" fill="${JEST_HAT}"/>
  <path d="M32 16 C41 17 49 13 55 6 C58 14 54 22 45 24 Z" fill="${JEST_HAT_DARK}"/>
  <path d="M32 17 C28.5 11 29.5 4 32 0.5 C34.5 4 35.5 11 32 17 Z" fill="${JEST_HAT_LIT}"/>
  <circle cx="9" cy="6" r="3.2" fill="${BELL}"/>
  <circle cx="55" cy="6" r="3.2" fill="${BELL}"/>
  <circle cx="32" cy="1.5" r="2.8" fill="${BELL}"/>
  <path d="M17 22 C23 16 41 16 47 22 L47 27 C41 21.5 23 21.5 17 27 Z" fill="${JEST_HAT_LIT}"/>
`;

const jesterCheeks = `
  <circle cx="21.5" cy="44" r="2.8" fill="#f4a0a8" opacity="0.75"/>
  <circle cx="42.5" cy="44" r="2.8" fill="#f4a0a8" opacity="0.75"/>
`;

export const JESTER_GLEEFUL = `
  ${jesterHat}
  <circle cx="32" cy="41" r="13" fill="${SKIN}"/>
  ${jesterCheeks}
  <path d="M23 39.5 C25.5 35.5 29 35.5 31 39.5" stroke="${INK}" stroke-width="2.1" fill="none" stroke-linecap="round"/>
  <path d="M33 39.5 C35 35.5 38.5 35.5 41 39.5" stroke="${INK}" stroke-width="2.1" fill="none" stroke-linecap="round"/>
  <path d="M23.5 45 C26.5 53.5 37.5 53.5 40.5 45 Z" fill="#7d2f45"/>
  <path d="M28.5 51.5 C29.5 55.5 34.5 55.5 35.5 51.5 Z" fill="#ff92ab"/>
`;

export const JESTER_STRAINING = `
  ${jesterHat}
  <circle cx="32" cy="41" r="13" fill="${SKIN}"/>
  ${jesterCheeks}
  <path d="M22.5 34.5 C25.5 36 28.5 37.5 30 39.5" stroke="${INK}" stroke-width="2.3" fill="none" stroke-linecap="round"/>
  <path d="M41.5 34.5 C38.5 36 35.5 37.5 34 39.5" stroke="${INK}" stroke-width="2.3" fill="none" stroke-linecap="round"/>
  <path d="M24.5 41.5 L30 41.5" stroke="${INK}" stroke-width="2.1" fill="none" stroke-linecap="round"/>
  <path d="M34 41.5 L39.5 41.5" stroke="${INK}" stroke-width="2.1" fill="none" stroke-linecap="round"/>
  <rect x="25" y="46.5" width="14" height="6.5" rx="2" fill="#7d2f45"/>
  <rect x="25" y="46.5" width="14" height="3" rx="1.4" fill="#ffffff"/>
  <path d="M29.5 46.5 L29.5 49.5 M34.5 46.5 L34.5 49.5" stroke="#c9b8bd" stroke-width="0.9"/>
  <path d="M50 30 C52 33.5 53 35.5 53 37 C53 39 51.5 40.2 50 40.2 C48.5 40.2 47 39 47 37 C47 35.5 48 33.5 50 30 Z" fill="#8fd4ff"/>
  <path d="M14 34 C15.6 36.8 16.4 38.4 16.4 39.6 C16.4 41.2 15.2 42.2 14 42.2 C12.8 42.2 11.6 41.2 11.6 39.6 C11.6 38.4 12.4 36.8 14 34 Z" fill="#8fd4ff"/>
`;

// --- Toy: little teaching robot ----------------------------------------

export const TOY_BOT = `
  <rect x="27.5" y="9" width="9" height="9" rx="2.5" fill="#0f9c8f"/>
  <circle cx="32" cy="6.5" r="3.4" fill="${BELL}"/>
  <rect x="6" y="30" width="7" height="13" rx="3.5" fill="#0f9c8f"/>
  <rect x="51" y="30" width="7" height="13" rx="3.5" fill="#0f9c8f"/>
  <rect x="13" y="17" width="38" height="34" rx="8" fill="#2dd4bf"/>
  <rect x="19" y="24" width="26" height="16" rx="4" fill="#08312e"/>
  <circle cx="26" cy="32" r="3.2" fill="#5ae3d8"/>
  <circle cx="38" cy="32" r="3.2" fill="#5ae3d8"/>
  <path d="M26 45 C29 47.5 35 47.5 38 45" stroke="#0b3b38" stroke-width="2.2" fill="none" stroke-linecap="round"/>
  <rect x="19" y="51" width="10" height="6" rx="2.4" fill="#0f9c8f"/>
  <rect x="35" y="51" width="10" height="6" rx="2.4" fill="#0f9c8f"/>
`;

export type BotIconName = 'sage-calm' | 'sage-flustered' | 'jester-gleeful' | 'jester-straining' | 'toy';

export const BOT_ICONS: Record<BotIconName, string> = {
  'sage-calm': SAGE_CALM,
  'sage-flustered': SAGE_FLUSTERED,
  'jester-gleeful': JESTER_GLEEFUL,
  'jester-straining': JESTER_STRAINING,
  toy: TOY_BOT,
};
