import numpy as np
import chess
import pandas as pd
import os
import struct
import tqdm
import argparse
from typing import List, Tuple, Dict

# Constants for halfKP feature calculation
SQUARE_NB = 64
PIECE_NB = 10  # All piece types except kings (5 types * 2 colors)
KING_SQUARE_NB = 64

# Feature index calculation constants
FEATURE_DIMENSIONS = (KING_SQUARE_NB, PIECE_NB, SQUARE_NB)

def get_halfkp_indices(board: chess.Board) -> Tuple[List[int], List[int]]:
    """
    Generate halfKP features for a given board position.
    Returns two lists of feature indices (white perspective and black perspective).
    
    HalfKP features: For each king position, for each piece type, for each square.
    """
    white_features = []
    black_features = []
    
    # Find king positions
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)
    
    if white_king_sq is None or black_king_sq is None:
        # Invalid position without kings
        return [], []
    
    # Map pieces to indices (0-9)
    # Order: pawn, knight, bishop, rook, queen for each color
    piece_idx_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4
    }
    
    # Process each square on the board
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type == chess.KING:
            continue  # Skip empty squares and kings
        
        piece_type = piece.piece_type
        color = piece.color
        
        # Get base piece index (0-4 for white pieces, 5-9 for black pieces)
        if color == chess.WHITE:
            piece_idx = piece_idx_map[piece_type]
        else:
            piece_idx = piece_idx_map[piece_type] + 5
        
        # White perspective feature
        white_feature_idx = white_king_sq * PIECE_NB * SQUARE_NB + piece_idx * SQUARE_NB + sq
        white_features.append(white_feature_idx)
        
        # Black perspective feature (with board mirroring)
        # Mirror the square and king square (flip rank)
        mirror_sq = (7 - sq // 8) * 8 + (sq % 8)
        mirror_king_sq = (7 - black_king_sq // 8) * 8 + (black_king_sq % 8)
        
        # Mirror piece color (white pieces are black from black's perspective)
        if color == chess.WHITE:
            mirror_piece_idx = piece_idx_map[piece_type] + 5  # Make it a black piece
        else:
            mirror_piece_idx = piece_idx_map[piece_type]  # Make it a white piece
        
        black_feature_idx = mirror_king_sq * PIECE_NB * SQUARE_NB + mirror_piece_idx * SQUARE_NB + mirror_sq
        black_features.append(black_feature_idx)
    
    return white_features, black_features

def save_features_binary(output_file: str, 
                         fens: List[str], 
                         white_features: List[List[int]], 
                         black_features: List[List[int]], 
                         scores: List[float]):
    """
    Save the features to a binary file for efficient storage.
    """
    with open(output_file, 'wb') as f:
        # Write header: number of positions
        f.write(struct.pack('i', len(fens)))
        
        # Write each position
        for i in range(len(fens)):
            # Write FEN
            fen_bytes = fens[i].encode('utf-8')
            f.write(struct.pack('i', len(fen_bytes)))
            f.write(fen_bytes)
            
            # Write score
            f.write(struct.pack('f', scores[i]))
            
            # Write white features
            f.write(struct.pack('i', len(white_features[i])))
            f.write(struct.pack(f'{len(white_features[i])}i', *white_features[i]))
            
            # Write black features
            f.write(struct.pack('i', len(black_features[i])))
            f.write(struct.pack(f'{len(black_features[i])}i', *black_features[i]))

def read_features_binary(input_file: str) -> Tuple[List[str], List[List[int]], List[List[int]], List[float]]:
    """
    Read feature data from binary file.
    """
    fens = []
    white_features = []
    black_features = []
    scores = []
    
    with open(input_file, 'rb') as f:
        # Read number of positions
        num_positions = struct.unpack('i', f.read(4))[0]
        
        for _ in range(num_positions):
            # Read FEN
            fen_length = struct.unpack('i', f.read(4))[0]
            fen = f.read(fen_length).decode('utf-8')
            fens.append(fen)
            
            # Read score
            score = struct.unpack('f', f.read(4))[0]
            scores.append(score)
            
            # Read white features
            num_white_features = struct.unpack('i', f.read(4))[0]
            white_feature_data = f.read(num_white_features * 4)
            white_feature_list = list(struct.unpack(f'{num_white_features}i', white_feature_data))
            white_features.append(white_feature_list)
            
            # Read black features
            num_black_features = struct.unpack('i', f.read(4))[0]
            black_feature_data = f.read(num_black_features * 4)
            black_feature_list = list(struct.unpack(f'{num_black_features}i', black_feature_data))
            black_features.append(black_feature_list)
    
    return fens, white_features, black_features, scores

def process_kaggle_dataset(input_file: str, output_file: str, max_positions: int = None):
    """
    Process the Kaggle chess dataset and convert to halfKP features.
    
    Args:
        input_file: Path to Kaggle dataset CSV file
        output_file: Path to save the features binary file
        max_positions: Maximum number of positions to process (None for all)
    """
    print(f"Reading dataset from {input_file}")
    df = pd.read_csv(input_file)
    
    # Validate dataset format - should have 'FEN' and 'Evaluation' columns
    required_columns = ['FEN', 'Evaluation']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")
    
    # Limit number of positions if specified
    if max_positions is not None:
        df = df.head(max_positions)
    
    num_positions = len(df)
    print(f"Processing {num_positions} positions")
    
    fens = []
    white_features_list = []
    black_features_list = []
    scores = []
    
    # Process each position
    for idx, row in tqdm.tqdm(df.iterrows(), total=num_positions):
        fen = row['FEN']
        # Convert centipawn evaluation to float value (pawn = 1.0)
        score = float(row['Evaluation']) / 100.0
        
        # Parse FEN
        try:
            board = chess.Board(fen)
            white_features, black_features = get_halfkp_indices(board)
            
            # Skip positions with no features (invalid positions)
            if len(white_features) == 0 or len(black_features) == 0:
                continue
                
            fens.append(fen)
            white_features_list.append(white_features)
            black_features_list.append(black_features)
            scores.append(score)
            
        except Exception as e:
            print(f"Error processing position {idx} (FEN: {fen}): {str(e)}")
    
    print(f"Successfully processed {len(fens)} valid positions")
    
    # Save features to binary file
    print(f"Saving features to {output_file}")
    save_features_binary(output_file, fens, white_features_list, black_features_list, scores)
    print(f"Data saved successfully")
    
    # Print some statistics
    feature_counts = [len(features) for features in white_features_list]
    print(f"Average features per position: {np.mean(feature_counts):.2f}")
    print(f"Min features: {min(feature_counts)}, Max features: {max(feature_counts)}")

def create_pytorch_dataset(input_file: str, output_dir: str):
    """
    Convert binary features file to PyTorch-ready dataset files.
    """
    import torch
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading features from {input_file}")
    fens, white_features, black_features, scores = read_features_binary(input_file)
    
    # Create sparse feature tensors and save in chunks
    chunk_size = 100000
    num_chunks = (len(fens) + chunk_size - 1) // chunk_size
    
    total_features = KING_SQUARE_NB * PIECE_NB * SQUARE_NB
    
    print(f"Converting to PyTorch format and saving in {num_chunks} chunks")
    for chunk in range(num_chunks):
        start_idx = chunk * chunk_size
        end_idx = min(start_idx + chunk_size, len(fens))
        
        # Prepare data for this chunk
        chunk_scores = torch.tensor([scores[i] for i in range(start_idx, end_idx)], dtype=torch.float32)
        
        # Create white perspective sparse tensors
        white_indices = []
        white_values = []
        for i in range(start_idx, end_idx):
            for feature_idx in white_features[i]:
                white_indices.append([i - start_idx, feature_idx])
                white_values.append(1.0)
        
        if white_indices:
            white_indices_tensor = torch.tensor(white_indices, dtype=torch.long).t()
            white_values_tensor = torch.tensor(white_values, dtype=torch.float32)
            white_sparse = torch.sparse_coo_tensor(
                white_indices_tensor, 
                white_values_tensor,
                size=(end_idx - start_idx, total_features)
            )
        else:
            # Empty tensor if no features
            white_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                size=(end_idx - start_idx, total_features)
            )
        
        # Create black perspective sparse tensors
        black_indices = []
        black_values = []
        for i in range(start_idx, end_idx):
            for feature_idx in black_features[i]:
                black_indices.append([i - start_idx, feature_idx])
                black_values.append(1.0)
        
        if black_indices:
            black_indices_tensor = torch.tensor(black_indices, dtype=torch.long).t()
            black_values_tensor = torch.tensor(black_values, dtype=torch.float32)
            black_sparse = torch.sparse_coo_tensor(
                black_indices_tensor, 
                black_values_tensor,
                size=(end_idx - start_idx, total_features)
            )
        else:
            # Empty tensor if no features
            black_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                size=(end_idx - start_idx, total_features)
            )
        
        # Save this chunk
        chunk_file = os.path.join(output_dir, f"chunk_{chunk}.pt")
        torch.save({
            'white_features': white_sparse,
            'black_features': black_sparse,
            'scores': chunk_scores
        }, chunk_file)
        
        print(f"Saved chunk {chunk+1}/{num_chunks}")
    
    # Save metadata
    metadata = {
        'num_chunks': num_chunks,
        'total_positions': len(fens),
        'feature_dimensions': FEATURE_DIMENSIONS,
        'total_features': total_features
    }
    
    metadata_file = os.path.join(output_dir, "metadata.pt")
    torch.save(metadata, metadata_file)
    print(f"Saved metadata to {metadata_file}")

# Example PyTorch training setup
def generate_training_code():
    """
    Generate example code for training an NNUE network using PyTorch.
    """
    code = """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

# NNUE Network Architecture
class NNUE(nn.Module):
    def __init__(self, input_size, l1_size=256, l2_size=32):
        super(NNUE, self).__init__()
        
        # Feature Transformer (FC1)
        self.fc1 = nn.Linear(input_size, l1_size)
        # Set bias to clipped ReLU range
        self.fc1.bias.data.fill_(0)
        
        # FC2 Layer
        self.fc2 = nn.Linear(2 * l1_size, l2_size)
        self.fc2.bias.data.fill_(0)
        
        # Output Layer
        self.fc3 = nn.Linear(l2_size, 1)
        self.fc3.bias.data.fill_(0)
    
    def forward(self, white_features, black_features):
        # Feature Transformer (Layer 1) with ClippedReLU(0, 127)
        white_layer1 = torch.clamp(self.fc1(white_features), 0, 127)
        black_layer1 = torch.clamp(self.fc1(black_features), 0, 127)
        
        # Concatenate perspectives
        combined = torch.cat([white_layer1, black_layer1], dim=1)
        
        # Layer 2 with ReLU
        layer2 = torch.relu(self.fc2(combined))
        
        # Output layer (no activation)
        output = self.fc3(layer2)
        
        return output

class ChessDataset(Dataset):
    def __init__(self, data_dir, chunk_idx):
        chunk_file = os.path.join(data_dir, f"chunk_{chunk_idx}.pt")
        self.data = torch.load(chunk_file)
        
        # Convert sparse tensors to dense if necessary
        if self.data['white_features'].is_sparse:
            self.white_features = self.data['white_features'].to_dense()
        else:
            self.white_features = self.data['white_features']
            
        if self.data['black_features'].is_sparse:
            self.black_features = self.data['black_features'].to_dense()
        else:
            self.black_features = self.data['black_features']
            
        self.scores = self.data['scores']
    
    def __len__(self):
        return len(self.scores)
    
    def __getitem__(self, idx):
        return {
            'white_features': self.white_features[idx],
            'black_features': self.black_features[idx],
            'score': self.scores[idx]
        }

def train_nnue(data_dir, output_model, epochs=10, batch_size=1024, lr=0.01):
    # Load metadata
    metadata = torch.load(os.path.join(data_dir, "metadata.pt"))
    num_chunks = metadata['num_chunks']
    total_features = metadata['total_features']
    
    # Create model
    model = NNUE(input_size=total_features)
    
    # MSE loss is typical for regression problems like evaluation
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        total_positions = 0
        
        # Process each chunk
        for chunk_idx in range(num_chunks):
            dataset = ChessDataset(data_dir, chunk_idx)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for batch in dataloader:
                white_features = batch['white_features']
                black_features = batch['black_features']
                scores = batch['score'].view(-1, 1)
                
                # Forward pass
                outputs = model(white_features, black_features)
                loss = criterion(outputs, scores)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(scores)
                total_positions += len(scores)
            
            print(f"Epoch {epoch+1}/{epochs}, Chunk {chunk_idx+1}/{num_chunks}, "
                  f"Avg Loss: {total_loss/total_positions:.6f}")
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': metadata
    }, output_model)
    
    print(f"Model saved to {output_model}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NNUE network')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with processed data chunks')
    parser.add_argument('--output_model', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    
    args = parser.parse_args()
    
    train_nnue(args.data_dir, args.output_model, args.epochs, args.batch_size, args.lr)
"""
    
    with open('nnue_training.py', 'w') as f:
        f.write(code)
    
    print("Generated example NNUE training code: nnue_training.py")

def export_weights_to_binary(model_file: str, output_file: str):
    """
    Export trained PyTorch model weights to binary format compatible with the C++ NNUE implementation.
    """
    import torch
    
    # Load trained model
    checkpoint = torch.load(model_file)
    state_dict = checkpoint['model_state_dict']
    metadata = checkpoint['metadata']
    
    # Extract dimensions
    feature_dimensions = metadata['feature_dimensions']
    total_features = metadata['total_features']
    
    # Get layer sizes from state dict
    input_size = total_features
    l1_size = state_dict['fc1.weight'].size(0)
    l2_size = state_dict['fc2.weight'].size(0)
    
    # Write to binary file
    with open(output_file, 'wb') as f:
        # Write dimensions: input_size, l1_size, l2_size
        dims = [input_size, l1_size, l2_size]
        f.write(struct.pack('3i', *dims))
        
        # Convert and write FC1 weights (int16)
        fc1_weights = state_dict['fc1.weight'].numpy().astype(np.int16)
        f.write(fc1_weights.tobytes())
        
        # Convert and write FC1 bias (int16)
        fc1_bias = state_dict['fc1.bias'].numpy().astype(np.int16)
        f.write(fc1_bias.tobytes())
        
        # Reshape and write FC2 weights (int16)
        # Note: Our C++ implementation expects the weights in a particular order
        fc2_weights = state_dict['fc2.weight'].numpy().astype(np.int16)
        f.write(fc2_weights.tobytes())
        
        # Write FC2 bias (int16)
        fc2_bias = state_dict['fc2.bias'].numpy().astype(np.int16)
        f.write(fc2_bias.tobytes())
        
        # Write FC3 weights (int16)
        fc3_weights = state_dict['fc3.weight'].view(-1).numpy().astype(np.int16)
        f.write(fc3_weights.tobytes())
        
        # Write FC3 bias (int16)
        fc3_bias = state_dict['fc3.bias'].numpy().astype(np.int16)
        f.write(fc3_bias.tobytes())
    
    print(f"Exported weights to {output_file}")
    print(f"Network architecture: {input_size} -> {l1_size} -> {l2_size} -> 1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process chess positions and convert to NNUE features')
    subparsers = parser.add_subparsers(dest='command')
    
    # Process data command
    process_parser = subparsers.add_parser('process', help='Process Kaggle dataset')
    process_parser.add_argument('--input', type=str, required=True, help='Input CSV file (Kaggle dataset)')
    process_parser.add_argument('--output', type=str, required=True, help='Output binary file for features')
    process_parser.add_argument('--max_positions', type=int, help='Maximum number of positions to process')
    
    # Create PyTorch dataset command
    pytorch_parser = subparsers.add_parser('prepare_pytorch', help='Prepare PyTorch datasets')
    pytorch_parser.add_argument('--input', type=str, required=True, help='Input binary file with features')
    pytorch_parser.add_argument('--output_dir', type=str, required=True, help='Output directory for PyTorch datasets')
    
    # Generate training code command
    code_parser = subparsers.add_parser('generate_code', help='Generate training code')
    
    # Export weights command
    export_parser = subparsers.add_parser('export_weights', help='Export weights to binary format')
    export_parser.add_argument('--model', type=str, required=True, help='PyTorch model file')
    export_parser.add_argument('--output', type=str, required=True, help='Output binary weights file')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        process_kaggle_dataset(args.input, args.output, args.max_positions)
    elif args.command == 'prepare_pytorch':
        create_pytorch_dataset(args.input, args.output_dir)
    elif args.command == 'generate_code':
        generate_training_code()
    elif args.command == 'export_weights':
        export_weights_to_binary(args.model, args.output)
    else:
        parser.print_help()
