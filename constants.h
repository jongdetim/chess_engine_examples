#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cstdint>

// Chess constants
constexpr int PIECE_TYPES = 6;  // pawn, knight, bishop, rook, queen, king
constexpr int PIECE_COLORS = 2; // white, black
constexpr int SQUARES = 64;
constexpr int KING_SQUARES = 64;

// Piece types
enum Piece { EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
enum Color { WHITE, BLACK };
enum Square {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    NO_SQUARE
};

// Search constants
constexpr int INF = 30000;
constexpr int MATE_SCORE = 29000;
constexpr int DRAW_SCORE = 0;
constexpr int MAX_DEPTH = 100;
constexpr int QUIESCE_MAX_DEPTH = 10;
constexpr int TT_SIZE = 1 << 20; // 1M entries

// Evaluation constants
constexpr int PAWN_VALUE = 100;
constexpr int KNIGHT_VALUE = 320;
constexpr int BISHOP_VALUE = 330;
constexpr int ROOK_VALUE = 500;
constexpr int QUEEN_VALUE = 900;
constexpr int KING_VALUE = 20000;

#endif // CONSTANTS_H
