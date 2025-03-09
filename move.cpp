#include "move.h"

std::string Move::toUci() const {
    if (isNull()) return "0000";
    
    std::string result;
    result += 'a' + (from % 8);
    result += '1' + (from / 8);
    result += 'a' + (to % 8);
    result += '1' + (to / 8);
    
    if (promotion != EMPTY) {
        char promotionChar = ' ';
        switch (promotion) {
            case KNIGHT: promotionChar = 'n'; break;
            case BISHOP: promotionChar = 'b'; break;
            case ROOK:   promotionChar = 'r'; break;
            case QUEEN:  promotionChar = 'q'; break;
            default: break;
        }
        result += promotionChar;
    }
    
    return result;
}

Move Move::fromUci(const std::string& uci) {
    if (uci.length() < 4) return Move();
    
    int fromFile = uci[0] - 'a';
    int fromRank = uci[1] - '1';
    int toFile = uci[2] - 'a';
    int toRank = uci[3] - '1';
    
    if (fromFile < 0 || fromFile > 7 || fromRank < 0 || fromRank > 7 ||
        toFile < 0 || toFile > 7 || toRank < 0 || toRank > 7) {
        return Move();
    }
    
    Square from = static_cast<Square>(fromRank * 8 + fromFile);
    Square to = static_cast<Square>(toRank * 8 + toFile);
    
    Piece promotion = EMPTY;
    if (uci.length() > 4) {
        switch (uci[4]) {
            case 'n': promotion = KNIGHT; break;
            case 'b': promotion = BISHOP; break;
            case 'r': promotion = ROOK; break;
            case 'q': promotion = QUEEN; break;
        }
    }
    
    return Move(from, to, promotion);
}
