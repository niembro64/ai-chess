// Inline SVG silhouettes for the six chess pieces, drawn on the standard
// 45x45 chess-set grid. Both colors share one shape; CSS colors the fill
// via currentColor and the contrasting edge via the --piece-outline
// custom property (see ChessBoard.vue).
//
// Why SVG and not the Unicode glyphs (U+2654-265F) we used before:
// glyph rendering is platform font territory — iOS force-renders U+265F
// as an EMOJI (it carries the Emoji property since Unicode 11) and draws
// the rest from a different fallback font than desktop, so the board
// looked completely different (and chaotic) on iPhone. Inline SVG is
// byte-identical everywhere.

import type { PieceType } from '@/types/chess';

// Each entry is the inner markup of a 45x45 viewBox svg. Shapes are
// simple filled silhouettes: shared plinth + piece-specific body.
const BASE = '<rect x="10.5" y="35" width="24" height="5" rx="2.4"/>';

export const PIECE_ICONS: Record<PieceType, string> = {
  pawn:
    '<circle cx="22.5" cy="13" r="5.4"/>' +
    '<rect x="16.6" y="18.6" width="11.8" height="2.8" rx="1.4"/>' +
    '<path d="M19.3 21.4 h6.4 c0 5.4 2.2 9.6 3.9 14.1 h-14.2 c1.7 -4.5 3.9 -8.7 3.9 -14.1 z"/>' +
    BASE,

  rook:
    '<path d="M12.5 8 h4.6 v3.6 h3.2 V8 h4.4 v3.6 h3.2 V8 h4.6 v7.4 l-2.6 2.6 v11.2 l3.1 4 H12 l3.1 -4 V18 l-2.6 -2.6 z"/>' +
    BASE,

  knight:
    '<path d="M14.6 35.6 c0 -7.4 1.8 -11.6 5.2 -15 c-2.6 1.1 -5.8 3.4 -7.6 2.1 c-1.7 -1.3 -2.5 -3.5 -1.4 -4.9 c1.8 -2.3 3.7 -4.8 5.8 -6.7 c1.7 -1.5 2 -3 2.4 -5 l2.7 2.1 c5.3 0.6 9.5 3.2 11.4 9.1 c1.3 4.2 1.5 10 1.5 18.3 z"/>' +
    '<path d="M21 7.4 l2.2 -4.4 l2.1 5.2 z"/>' +
    BASE,

  bishop:
    '<circle cx="22.5" cy="7.8" r="2.6"/>' +
    '<path d="M22.5 10.6 c4.7 3.1 7.3 7.8 7.3 12.4 c0 3.6 -2 6.2 -3.6 7.7 h-7.4 c-1.6 -1.5 -3.6 -4.1 -3.6 -7.7 c0 -4.6 2.6 -9.3 7.3 -12.4 z"/>' +
    '<rect x="15.9" y="31.4" width="13.2" height="2.8" rx="1.4"/>' +
    BASE,

  queen:
    '<circle cx="11" cy="11.6" r="2.1"/>' +
    '<circle cx="22.5" cy="8.8" r="2.1"/>' +
    '<circle cx="34" cy="11.6" r="2.1"/>' +
    '<path d="M10.8 13.6 l4.8 13.4 h13.8 l4.8 -13.4 -6.1 6.7 -3.5 -9.2 h-4.2 l-3.5 9.2 z"/>' +
    '<path d="M15.6 27 c-0.9 3.6 -2.3 6 -3.7 8 h21.2 c-1.4 -2 -2.8 -4.4 -3.7 -8 z"/>' +
    BASE,

  king:
    '<rect x="21.1" y="2.8" width="2.8" height="8.4" rx="0.9"/>' +
    '<rect x="18.3" y="5.4" width="8.4" height="2.8" rx="0.9"/>' +
    '<path d="M22.5 11.2 c4.8 0 8 3 8 6.6 c0 3.4 -2.6 5.7 -5 6.7 h-6 c-2.4 -1 -5 -3.3 -5 -6.7 c0 -3.6 3.2 -6.6 8 -6.6 z"/>' +
    '<path d="M17.6 23.6 h9.8 l1.4 3.4 c0.9 3.4 2.2 6 3.6 8 H12.6 c1.4 -2 2.7 -4.6 3.6 -8 z"/>' +
    BASE,
};
