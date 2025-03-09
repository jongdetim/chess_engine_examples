#ifndef MOVE_H
#define MOVE_H

#include "constants.h"
#include <string>

// Move structure
class Move {
public:
    Square from;
    Square to;
    Piece promotion;
    
    Move() : from(NO_SQUARE), to(NO_SQUARE), promotion(EMPTY) {}
    Move(Square f, Square t, Piece p = EMPTY) : from(f), to(t), promotion(p) {}
    
    bool operator==(const Move& other) const {
        return from == other.from && to == other.to && promotion == other.promotion;
    }
    
    bool isNull() const {
        return from == NO_SQUARE || to == NO_SQUARE;
    }
    
    // Convert move to UCI string notation (e.g., "e2e4")
    std::string toUci() const;
    
    // Create move from UCI string
    static Move fromUci(const std::string& uci);
};

#endif // MOVE_H
