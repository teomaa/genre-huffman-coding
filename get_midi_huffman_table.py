#!/usr/bin/env python3
"""
Script to compress MIDI piano roll data using Huffman coding
and compare compressed vs uncompressed sizes.
"""

import argparse
import sys
import numpy as np
import pretty_midi
from collections import Counter
import heapq
from typing import Dict, Tuple, Optional


class HuffmanNode:
    """Node for Huffman tree"""
    def __init__(self, value: Optional[int] = None, freq: int = 0, left=None, right=None):
        self.value = value
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(data: np.ndarray) -> Tuple[HuffmanNode, Dict[int, str]]:
    """Build Huffman tree and return root node and encoding dictionary"""
    # Count frequencies
    flat_data = data.flatten()
    frequencies = Counter(flat_data.tolist())
    
    # Build priority queue
    heap = []
    for value, freq in frequencies.items():
        node = HuffmanNode(value=value, freq=freq)
        heapq.heappush(heap, node)
    
    # Build tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    root = heap[0] if heap else None
    
    # Build encoding dictionary
    encoding_dict = {}
    
    def traverse(node: HuffmanNode, code: str = ""):
        if node.value is not None:
            encoding_dict[node.value] = code
        else:
            if node.left:
                traverse(node.left, code + "0")
            if node.right:
                traverse(node.right, code + "1")
    
    if root:
        traverse(root)
    
    return root, encoding_dict


def encode_data(data: np.ndarray, encoding_dict: Dict[int, str]) -> str:
    """Encode data using Huffman encoding dictionary"""
    flat_data = data.flatten()
    encoded_bits = ''.join(encoding_dict[value] for value in flat_data)
    return encoded_bits


def decode_data(encoded_bits: str, root: HuffmanNode, expected_length: int, dtype) -> np.ndarray:
    """Decode Huffman-encoded bits back to original data"""
    decoded_values = []
    current_node = root
    
    # Handle edge case: if root is a leaf (only one unique value)
    if root.value is not None:
        # All values are the same, code is empty string
        for _ in range(expected_length):
            decoded_values.append(root.value)
    else:
        # Normal case: traverse the tree
        for bit in encoded_bits:
            if bit == '0':
                current_node = current_node.left
            else:  # bit == '1'
                current_node = current_node.right
            
            # If we've reached a leaf node, we found a value
            if current_node.value is not None:
                decoded_values.append(current_node.value)
                current_node = root  # Reset to root for next value
    
    # Verify we decoded the expected number of values
    if len(decoded_values) != expected_length:
        raise ValueError(
            f"Decoded {len(decoded_values)} values but expected {expected_length}"
        )
    
    return np.array(decoded_values, dtype=dtype)


def calculate_compressed_size(encoded_bits: str, encoding_dict: Dict[int, str]) -> int:
    """Calculate size of compressed data in bits"""
    # Size of encoded data
    data_size = len(encoded_bits)
    
    # Size of encoding dictionary (approximate)
    # For each entry: value (assuming 4 bytes for int) + code length (1 byte) + code bits
    dict_size = 0
    for _value, code in encoding_dict.items():
        dict_size += 4  # value (int)
        dict_size += 1  # code length (byte)
        dict_size += len(code)  # code bits (stored as bits, but we'll count bytes)
    
    # Convert bits to bytes (round up)
    total_bits = data_size + dict_size * 8
    total_bytes = (total_bits + 7) // 8
    
    return total_bytes


def main():
    parser = argparse.ArgumentParser(
        description='Compress MIDI piano roll using Huffman coding'
    )
    parser.add_argument(
        'midi_path',
        type=str,
        help='Path to the MIDI file'
    )
    
    args = parser.parse_args()
    
    try:
        # Load MIDI file
        print(f"Loading MIDI file: {args.midi_path}")
        midi_data = pretty_midi.PrettyMIDI(args.midi_path)
        
        # Get piano roll
        print("Extracting piano roll...")
        piano_roll = midi_data.get_piano_roll()

        
        # Get original size (in bytes)
        original_size = piano_roll.nbytes
        print(f"Original piano roll size: {original_size:,} bytes")
        print(f"Piano roll shape: {piano_roll.shape}")
        
        # Build Huffman tree and get encoding
        print("Building Huffman tree...")
        root, encoding_dict = build_huffman_tree(piano_roll)
        
        if not encoding_dict or root is None:
            print("Error: Could not build encoding dictionary")
            sys.exit(1)
        
        print(f"Number of unique values: {len(encoding_dict)}")
        
        # Encode data
        print("Encoding data...")
        encoded_bits = encode_data(piano_roll, encoding_dict)
        
        # Decode and verify correctness
        print("Decoding data to verify compression...")
        original_shape = piano_roll.shape
        original_flat = piano_roll.flatten()
        decoded_flat = decode_data(encoded_bits, root, len(original_flat), piano_roll.dtype)
        
        # Compare original with decoded
        if not np.array_equal(original_flat, decoded_flat):
            print("\n" + "="*50)
            print("ERROR: DECOMPRESSION VERIFICATION FAILED!")
            print("="*50)
            print(f"Original shape: {original_shape}")
            print(f"Decoded shape: {decoded_flat.shape}")
            
            # Find differences
            differences = np.where(original_flat != decoded_flat)[0]
            print(f"Number of mismatches: {len(differences)}")
            if len(differences) > 0:
                print("First 10 mismatches:")
                for idx in differences[:10]:
                    print(f"  Index {idx}: original={original_flat[idx]}, decoded={decoded_flat[idx]}")
            
            raise ValueError("Decompressed data does not match original data!")
        
        print("✓ Verification passed: Decompressed data matches original exactly")
        
        # Calculate compressed size
        compressed_size = calculate_compressed_size(encoded_bits, encoding_dict)
        print(f"Compressed size (including tree): {compressed_size:,} bytes")
        
        # Calculate compression ratio
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        print("\n" + "="*50)
        print("COMPRESSION RESULTS")
        print("="*50)
        print(f"Original size:     {original_size:,} bytes")
        print(f"Compressed size:   {compressed_size:,} bytes")
        print(f"Compression ratio: {compression_ratio:.2f}%")
        print("="*50)
        
        if compression_ratio > 0:
            print(f"✓ Compression achieved: {compression_ratio:.2f}% reduction")
        else:
            print(f"✗ No compression: {abs(compression_ratio):.2f}% overhead")
        
    except FileNotFoundError:
        print(f"Error: MIDI file not found: {args.midi_path}")
        sys.exit(1)
    except (ValueError, IOError, OSError) as e:
        print(f"Error processing MIDI file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

