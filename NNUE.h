#ifndef NNUE_H
#define NNUE_H

#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include "chess_board.h"

// Forward declarations
class ChessBoard;

// NNUE weights structure
class NNUEWeights {
public:
    int input_size;
    int l1_size;
    int l2_size;
    
    std::vector<int16_t> fc1_weights;
    std::vector<int16_t> fc1_bias;
    std::vector<int16_t> fc2_weights;
    std::vector<int16_t> fc2_bias;
    std::vector<int16_t> fc3_weights;
    std::vector<int16_t> fc3_bias;
    
    NNUEWeights(const std::string& filename);
};

// NNUE evaluator class
class NNUEEvaluator {
private:
    const NNUEWeights& weights;
    
    // Temporary buffers for computation
    std::vector<int16_t> l1_output_white;
    std::vector<int16_t> l1_output_black;
    std::vector<int16_t> l2_output;
    
    // SIMD optimized matrix multiplication
    void forwardPassLayer1(const std::vector<int>& activeFeatures, std::vector<int16_t>& output);
    void forwardPassLayer2(std::vector<int16_t>& output);
    int forwardPassLayer3();
    
    // Helper for SIMD
    bool useAVX2;
    
public:
    NNUEEvaluator(const NNUEWeights& w);
    
    // Evaluate a position
    int evaluate(const ChessBoard& board);
    
    // Reset the accumulator
    void resetAccumulator();
};

// Singleton NNUE evaluator
class NNUE {
private:
    static std::unique_ptr<NNUEWeights> weights;
    static std::unique_ptr<NNUEEvaluator> evaluator;
    
public:
    static void init(const std::string& weightsFile);
    static int evaluate(const ChessBoard& board);
    static bool isInitialized() { return evaluator != nullptr; }
};

#endif // NNUE_H
