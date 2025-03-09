#ifndef CHESS_BOARD_H
#define CHESS_BOARD_H

#include "constants.h"
#include "types.h"
#include "move.h"
#include "bitboard.h"
#include <string>
#include <vector>
#include <array>

// Board state for undo move
struct BoardState {
    std::array<std::array<Bitboard, PIECE_TYPES + 1>, PIECE_COLORS> pieces;
    Color sideToMove;
    std::array<std::array<bool, 2>, PIECE_COLORS> castlingRights; // [color][kingside/queenside]
    Square enPassantSquare;
    int halfmoveClock;
    int fullmoveNumber;
    std::array<Square, PIECE_COLORS> kingSquare;
    uint64_t zobristHash;
    Move lastMove;
};

// Chess board representation with bitboards
class ChessBoard {
private:
    // Bitboards for each piece type and color
    std::array<std::array<Bitboard, PIECE_TYPES + 1>, PIECE_COLORS> pieces; // +1 for an "all pieces" bitboard
    Color sideToMove;
    
    // Castling rights [color][kingside/queenside]
    std::array<std::array<bool, 2>, PIECE_COLORS> castlingRights;
    
    // En passant square
    Square enPassantSquare;
    
    // Halfmove clock and fullmove number
    int halfmoveClock;
    int fullmoveNumber;
    
    // Move history for undo
    std::vector<BoardState> history;
    
    // King square cache
    std::array<Square, PIECE_COLORS> kingSquare;
    
    // Zobrist hash
    uint64_t zobristHash;
    static std::array<std::array<std::array<uint64_t, SQUARES>, PIECE_TYPES + 1>, PIECE_COLORS> zobristKeys; // [color][piece][square]
    static uint64_t zobristSideToMove;
    static std::array<std::array<uint64_t, 2>, PIECE_COLORS> zobristCastling; // [color][kingside/queenside]
    static std::array<uint64_t, SQUARES> zobristEnPassant;
    static bool zobristInitialized;
    
    // NNUE-related state
    std::vector<int> whiteActiveFeatures;
    std::vector<int> blackActiveFeatures;
    bool nnueDirty;
    
    // Helper methods
    void initZobristKeys();
    void updateZobristHash();
    Piece pieceTypeAt(Square sq) const;
    
    // Check if a square is attacked
    bool isSquareAttacked(Square sq, Color attacker) const;
    
    // Generate all pseudo-legal moves
    std::vector<Move> generatePseudoLegalMoves() const;
    
    // Filter out illegal moves from pseudo-legal ones
    std::vector<Move> filterLegalMoves(const std::vector<Move>& pseudoLegalMoves) const;

public:
    ChessBoard();
    
    // FEN parsing
    void setFromFEN(const std::string& fen);
    std::string toFEN() const;
    
    // Move generation
    std::vector<Move> generateLegalMoves() const;
    bool isLegalMove(const Move& move) const;
    
    // Game state checks
    bool isCheck() const;
    bool isCheckmate() const;
    bool isStalemate() const;
    bool isDraw() const; // Includes stalemate, 50-move rule, repetition, insufficient material
    
    // Make/unmake move
    bool makeMove(const Move& move);
    void unmakeMove();
    
    // Board utility functions
    Piece pieceAt(Square sq) const;
    Color colorAt(Square sq) const;
    bool isEmpty(Square sq) const;
    Color getSideToMove() const { return sideToMove; }
    Square getKingSquare(Color color) const { return kingSquare[color]; }
    uint64_t getZobristHash() const { return zobristHash; }
    
    // NNUE-related functions
    void updateNNUEFeatures();
    int halfkpIndex(Square kingSquare, Piece pieceType, Color pieceColor, Square pieceSquare) const;
    const std::vector<int>& getWhiteFeatures() const { return whiteActiveFeatures; }
    const std::vector<int>& getBlackFeatures() const { return blackActiveFeatures; }
    
    // Debugging
    void printBoard() const;
};

#endif // CHESS_BOARD_H
