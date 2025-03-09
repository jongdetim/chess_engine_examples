#ifndef BITBOARD_H
#define BITBOARD_H

#include "types.h"
#include "constants.h"
#include <string>
#include <array>

namespace Bitboards {
    // Initialize lookup tables for move generation
    void init();
    
    // Bit manipulation functions
    inline Bitboard setBit(Bitboard b, Square s) { return b | (1ULL << s); }
    inline Bitboard clearBit(Bitboard b, Square s) { return b & ~(1ULL << s); }
    inline bool testBit(Bitboard b, Square s) { return (b & (1ULL << s)) != 0; }
    
    // Precomputed attack tables
    extern std::array<Bitboard, SQUARES> KnightAttacks;
    extern std::array<Bitboard, SQUARES> KingAttacks;
    extern std::array<std::array<Bitboard, SQUARES>, PIECE_COLORS> PawnAttacks;
    
    // Magic bitboard tables for sliding pieces
    extern std::array<std::array<Bitboard, 512>, SQUARES> BishopAttacks;
    extern std::array<std::array<Bitboard, 4096>, SQUARES> RookAttacks;
    
    // Get attacks for each piece type
    Bitboard getPawnAttacks(Square sq, Color color);
    Bitboard getKnightAttacks(Square sq);
    Bitboard getBishopAttacks(Square sq, Bitboard occupied);
    Bitboard getRookAttacks(Square sq, Bitboard occupied);
    Bitboard getQueenAttacks(Square sq, Bitboard occupied);
    Bitboard getKingAttacks(Square sq);
    
    // Utility functions
    int popCount(Bitboard b);
    Square lsb(Bitboard b);  // Least significant bit (first set bit)
    Square popLsb(Bitboard& b); // Return and clear least significant bit
    std::string toString(Bitboard b); // Convert bitboard to string for display
}

#endif // BITBOARD_H
