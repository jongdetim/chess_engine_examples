#ifndef TYPES_H
#define TYPES_H

#include "constants.h"
#include <cstdint>

// Bitboard representation
using Bitboard = uint64_t;

// Transposition table entry flags
enum TTFlag {
    TT_EXACT,
    TT_LOWER_BOUND,
    TT_UPPER_BOUND
};

#endif // TYPES_H
