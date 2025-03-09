#include "bitboard.h"
#include <sstream>
#include <iomanip>

namespace Bitboards {
    // Attack tables
    std::array<Bitboard, SQUARES> KnightAttacks;
    std::array<Bitboard, SQUARES> KingAttacks;
    std::array<std::array<Bitboard, SQUARES>, PIECE_COLORS> PawnAttacks;
    
    // Magic bitboard tables
    std::array<std::array<Bitboard, 512>, SQUARES> BishopAttacks;
    std::array<std::array<Bitboard, 4096>, SQUARES> RookAttacks;
    
    // Magic numbers and shift values for bishop and rook attacks
    const std::array<Bitboard, SQUARES> BishopMagics = { /* Magic numbers would be here */ };
    const std::array<Bitboard, SQUARES> RookMagics = { /* Magic numbers would be here */ };
    const std::array<int, SQUARES> BishopShifts = { /* Shift values would be here */ };
    const std::array<int, SQUARES> RookShifts = { /* Shift values would be here */ };
    
    // Init all attack tables - typically called at program startup
    void init() {
        // Init knight attacks
        for (Square sq = A1; sq < NO_SQUARE; ++sq) {
            Bitboard b = 0ULL;
            int r = sq / 8;
            int f = sq % 8;
            
            const int dr[] = {-2, -2, -1, -1, 1, 1, 2, 2};
            const int df[] = {-1, 1, -2, 2, -2, 2, -1, 1};
            
            for (int i = 0; i < 8; ++i) {
                int newR = r + dr[i];
                int newF = f + df[i];
                if (newR >= 0 && newR < 8 && newF >= 0 && newF < 8) {
                    b |= 1ULL << (newR * 8 + newF);
                }
            }
            
            KnightAttacks[sq] = b;
        }
        
        // Init king attacks
        for (Square sq = A1; sq < NO_SQUARE; ++sq) {
            Bitboard b = 0ULL;
            int r = sq / 8;
            int f = sq % 8;
            
            for (int dr = -1; dr <= 1; ++dr) {
                for (int df = -1; df <= 1; ++df) {
                    if (dr == 0 && df == 0) continue;
                    
                    int newR = r + dr;
                    int newF = f + df;
                    if (newR >= 0 && newR < 8 && newF >= 0 && newF < 8) {
                        b |= 1ULL << (newR * 8 + newF);
                    }
                }
            }
            
            KingAttacks[sq] = b;
        }
        
        // Init pawn attacks
        for (Square sq = A1; sq < NO_SQUARE; ++sq) {
            int r = sq / 8;
            int f = sq % 8;
            
            // White pawns attack up and diagonally
            Bitboard wAttacks = 0ULL;
            if (r < 7) {
                if (f > 0) wAttacks |= 1ULL << (sq + 7);
                if (f < 7) wAttacks |= 1ULL << (sq + 9);
            }
            PawnAttacks[WHITE][sq] = wAttacks;
            
            // Black pawns attack down and diagonally
            Bitboard bAttacks = 0ULL;
            if (r > 0) {
                if (f > 0) bAttacks |= 1ULL << (sq - 9);
                if (f < 7) bAttacks |= 1ULL << (sq - 7);
            }
            PawnAttacks[BLACK][sq] = bAttacks;
        }
        
        // Init bishop and rook attack tables (magic bitboards)
        // This would be a complex initialization for the magic bitboard tables
        // For brevity, we'll assume this is implemented elsewhere
    }
    
    // Get pawn attacks
    Bitboard getPawnAttacks(Square sq, Color color) {
        return PawnAttacks[color][sq];
    }
    
    // Get knight attacks
    Bitboard getKnightAttacks(Square sq) {
        return KnightAttacks[sq];
    }
    
    // Get bishop attacks using magic bitboards
    Bitboard getBishopAttacks(Square sq, Bitboard occupied) {
        // For a real implementation, this would use magic bitboards
        // Here's a simplified version assuming the tables are initialized
        Bitboard blockers = occupied & /* bishop mask for square */;
        int index = (blockers * BishopMagics[sq]) >> BishopShifts[sq];
        return BishopAttacks[sq][index];
    }
    
    // Get rook attacks using magic bitboards
    Bitboard getRookAttacks(Square sq, Bitboard occupied) {
        // For a real implementation, this would use magic bitboards
        // Here's a simplified version assuming the tables are initialized
        Bitboard blockers = occupied & /* rook mask for square */;
        int index = (blockers * RookMagics[sq]) >> RookShifts[sq];
        return RookAttacks[sq][index];
    }
    
    // Get queen attacks (bishop + rook)
    Bitboard getQueenAttacks(Square sq, Bitboard occupied) {
        return getBishopAttacks(sq, occupied) | getRookAttacks(sq, occupied);
    }
    
    // Get king attacks
    Bitboard getKingAttacks(Square sq) {
        return KingAttacks[sq];
    }
    
    // Count set bits in a bitboard
    int popCount(Bitboard b) {
        #if defined(__GNUC__) || defined(__clang__)
            // Use built-in function if available
            return __builtin_popcountll(b);
        #else
            // Fallback implementation
            int count = 0;
            while (b) {
                count++;
                b &= b - 1; // Clear least significant bit
            }
            return count;
        #endif
    }
    
    // Get index of least significant bit
    Square lsb(Bitboard b) {
        if (b == 0) return NO_SQUARE;
        
        #if defined(__GNUC__) || defined(__clang__)
            // Use built-in function if available
            return static_cast<Square>(__builtin_ctzll(b));
        #else
            // Fallback De Bruijn multiplication
            static const int index64[64] = {
                0,  1,  2,  7,  3, 13,  8, 19,
                4, 25, 14, 28,  9, 34, 20, 40,
                5, 17, 26, 38, 15, 46, 29, 48,
                10, 31, 35, 54, 21, 50, 41, 57,
                63,  6, 12, 18, 24, 27, 33, 39,
                16, 37, 45, 47, 30, 53, 49, 56,
                62, 11, 23, 32, 36, 44, 52, 55,
                61, 22, 43, 51, 60, 42, 59, 58
            };
            const Bitboard debruijn64 = 0x03f79d71b4cb0a89ULL;
            return static_cast<Square>(index64[((b & -b) * debruijn64) >> 58]);
        #endif
    }
    
    // Pop least significant bit
    Square popLsb(Bitboard& b) {
        Square s = lsb(b);
        b &= b - 1; // Clear least significant bit
        return s;
    }
    
    // Convert bitboard to string for display
    std::string toString(Bitboard b) {
        std::stringstream ss;
        for (int rank = 7; rank >= 0; --rank) {
            ss << (rank + 1) << " ";
            for (int file = 0; file < 8; ++file) {
                Square sq = static_cast<Square>(rank * 8 + file);
                ss << (testBit(b, sq) ? "1 " : ". ");
            }
            ss << "\n";
        }
        ss << "  a b c d e f g h";
        return ss.str();
    }
}
