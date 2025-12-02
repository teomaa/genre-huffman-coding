#!/usr/bin/env python3
"""
Script to build a genre-wide Huffman code from all MIDI files in a directory
and analyze compression ratios for individual songs.

Symbols are intervals between successive notes: pitch[i+1] - pitch[i]
This captures melodic contour and is transposition-invariant.
Intervals are computed within each song (reset after each song).
"""

import argparse
import sys
import numpy as np
import pretty_midi
from collections import Counter
import heapq
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm


class HuffmanNode:
    """Node for Huffman tree"""
    def __init__(self, value: Optional[int] = None, freq: int = 0, left=None, right=None):
        self.value = value
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(intervals: np.ndarray) -> Tuple[HuffmanNode, Dict[int, str]]:
    """Build Huffman tree and return root node and encoding dictionary"""
    # Count frequencies of intervals
    frequencies = Counter(intervals.tolist())
    
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


def encode_data(intervals: np.ndarray, encoding_dict: Dict[int, str]) -> str:
    """Encode interval data using Huffman encoding dictionary"""
    encoded_bits = ''.join(encoding_dict[interval] for interval in intervals)
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


def find_midi_files(directory: str) -> List[Tuple[str, str, str, str]]:
    """
    Find all MIDI files in a directory.
    Supports two structures:
    1. Single genre: genre/artist/song.mid
    2. Subgenre: genre/subgenre/artist/song.mid (flattens subgenres)
    
    Returns: List of (file_path, genre, artist, song_name) tuples
    Raises ValueError if directory structure is invalid
    """
    midi_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Get genre name from the directory name
    genre_name = directory_path.name
    
    # Find all .mid and .midi files in subdirectories
    for ext in ['*.mid', '*.midi']:
        for midi_file in directory_path.rglob(ext):
            midi_path = Path(midi_file)
            relative_path = midi_path.relative_to(directory_path)
            
            # Count depth: number of parent directories
            depth = len(relative_path.parts) - 1  # -1 because the file itself is a part
            
            if depth == 1:
                # Structure: genre/artist/song.mid (single genre)
                artist_name = relative_path.parent.name
                song_name = midi_path.stem
            elif depth == 2:
                # Structure: genre/subgenre/artist/song.mid (subgenre - flatten it)
                # Ignore subgenre name, use artist from the deepest level
                artist_name = relative_path.parent.name
                song_name = midi_path.stem
            else:
                # Invalid structure
                raise ValueError(
                    f"Invalid directory structure. Found MIDI file at depth {depth}: {midi_file}\n"
                    f"Expected structures:\n"
                    f"  - genre/artist/song.mid (depth 1)\n"
                    f"  - genre/subgenre/artist/song.mid (depth 2)"
                )
            
            midi_files.append((str(midi_file), genre_name, artist_name, song_name))
    
    if len(midi_files) == 0:
        raise ValueError(f"No MIDI files found in directory: {directory}")
    
    # Sort by genre, artist, then song
    midi_files.sort(key=lambda x: (x[1], x[2], x[3]))
    
    return midi_files


def extract_all_genre_intervals(directory: str) -> Tuple[np.ndarray, List[Tuple[str, str, str, str, np.ndarray]]]:
    """
    Extract intervals from all MIDI files in directory.
    Returns: (all_intervals, list of (file_path, genre, artist, song, intervals) for each file)
    Intervals are computed within each song (reset after each song).
    """
    midi_files = find_midi_files(directory)
    
    if len(midi_files) == 0:
        raise ValueError(f"No MIDI files found in directory: {directory}")
    
    all_intervals = []
    file_intervals = []
    
    print(f"Found {len(midi_files)} MIDI files")
    print("Extracting intervals from all files...")
    
    # Use tqdm for progress bar
    for file_path, genre, artist, song in tqdm(midi_files, desc="Processing MIDI files", unit="file"):
        try:
            midi_data = pretty_midi.PrettyMIDI(file_path)
            intervals = extract_note_intervals(midi_data)
            
            if len(intervals) > 0:
                all_intervals.extend(intervals.tolist())
                file_intervals.append((file_path, genre, artist, song, intervals))
        except Exception:  # noqa: BLE001
            # Continue processing other files if one fails
            # Catch all exceptions including MIDI parsing errors (KeySignatureError, etc.)
            # from mido/pretty_midi libraries, which may raise various exception types
            continue
    
    if len(all_intervals) == 0:
        raise ValueError("No intervals could be extracted from any MIDI files")
    
    return np.array(all_intervals, dtype=np.int32), file_intervals


def extract_note_intervals(midi_data: pretty_midi.PrettyMIDI) -> np.ndarray:
    """
    Extract note pitches from MIDI file and compute intervals between successive notes.
    Returns array of intervals: pitch[i+1] - pitch[i]
    """
    # Collect all notes from all instruments
    all_notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            all_notes.append((note.start, note.pitch))
    
    if len(all_notes) < 2:
        raise ValueError("MIDI file must contain at least 2 notes to compute intervals")
    
    # Sort notes by start time
    all_notes.sort(key=lambda x: x[0])
    
    # Extract pitches in chronological order
    pitches = np.array([note[1] for note in all_notes], dtype=np.int32)
    
    # Compute intervals: pitch[i+1] - pitch[i]
    intervals = np.diff(pitches)
    
    return intervals


def print_huffman_code(encoding_dict: Dict[int, str]):
    """Print the Huffman code in a readable format"""
    print("\n" + "="*50)
    print("GENRE HUFFMAN CODE")
    print("="*50)
    
    # Sort by code length, then by value
    sorted_codes = sorted(encoding_dict.items(), key=lambda x: (len(x[1]), x[0]))
    
    print(f"{'Interval':<12} {'Code':<20} {'Length'}")
    print("-" * 50)
    for interval, code in sorted_codes:
        sign = "+" if interval >= 0 else ""
        print(f"{sign}{interval:<11} {code:<20} {len(code)}")
    
    print("="*50)


def compress_file_with_code(intervals: np.ndarray, encoding_dict: Dict[int, str], genre: str, artist: str, song: str) -> float:
    """
    Compress a single file's intervals using the provided Huffman code.
    Returns compression ratio as a percentage.
    Note: Dictionary size is not included since it's shared across all files.
    """
    if len(intervals) == 0:
        return 0.0
    
    # Check if all intervals in this file are in the encoding dictionary
    missing_intervals = set(intervals) - set(encoding_dict.keys())
    if missing_intervals:
        # This shouldn't happen if we built the code from all files, but handle it
        print(f"  Warning: {len(missing_intervals)} interval values not in code for [{genre}] {artist} - {song}")
        return 0.0
    
    # Encode using the genre code
    encoded_bits = encode_data(intervals, encoding_dict)
    
    # Calculate sizes
    # Original: size of interval array in bytes
    original_size = intervals.nbytes
    
    # Compressed: only the encoded bits (dictionary is shared, not counted per file)
    # Convert bits to bytes (round up)
    compressed_size = (len(encoded_bits) + 7) // 8
    
    # Compression ratio (positive = compression achieved)
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    return compression_ratio


def main():
    parser = argparse.ArgumentParser(
        description='Build genre-wide Huffman code and analyze individual song compression'
    )
    parser.add_argument(
        'directory',
        type=str,
        help='Path to directory containing MIDI files'
    )
    
    args = parser.parse_args()
    
    try:
        # Extract intervals from all files in directory
        print(f"Processing directory: {args.directory}")
        all_intervals, file_intervals = extract_all_genre_intervals(args.directory)
        
        print(f"\nTotal intervals across all files: {len(all_intervals):,}")
        print(f"Interval range: [{all_intervals.min()}, {all_intervals.max()}]")
        
        # Build genre-wide Huffman tree
        print("\nBuilding genre-wide Huffman tree...")
        root, encoding_dict = build_huffman_tree(all_intervals)
        
        if not encoding_dict or root is None:
            print("Error: Could not build encoding dictionary")
            sys.exit(1)
        
        print(f"Number of unique interval values: {len(encoding_dict)}")
        
        # Print the Huffman code
        print_huffman_code(encoding_dict)
        
        # Compress each individual file using the genre code
        print("\n" + "="*50)
        print("COMPRESSING INDIVIDUAL FILES")
        print("="*50)
        
        compression_results = []
        
        for _file_path, genre, artist, song, intervals in tqdm(file_intervals, desc="Compressing files", unit="file"):
            compression_ratio = compress_file_with_code(intervals, encoding_dict, genre, artist, song)
            compression_results.append((genre, artist, song, compression_ratio, len(intervals)))
        
        # Find highest and lowest compression ratios
        if len(compression_results) == 0:
            print("No files were successfully processed")
            sys.exit(1)
        
        highest = max(compression_results, key=lambda x: x[3])
        lowest = min(compression_results, key=lambda x: x[3])
        
        # Calculate average compression ratio
        avg_compression = sum(result[3] for result in compression_results) / len(compression_results)
        
        print("\n" + "="*50)
        print("COMPRESSION SUMMARY")
        print("="*50)
        print(f"Highest compression: [{highest[0]}] {highest[1]} - {highest[2]}")
        print(f"  Compression ratio: {highest[3]:.2f}%")
        print(f"  Intervals: {highest[4]:,}")
        print()
        print(f"Lowest compression: [{lowest[0]}] {lowest[1]} - {lowest[2]}")
        print(f"  Compression ratio: {lowest[3]:.2f}%")
        print(f"  Intervals: {lowest[4]:,}")
        print()
        print(f"Average compression ratio: {avg_compression:.2f}%")
        print(f"  (across {len(compression_results)} files)")
        print("="*50)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except (IOError, OSError) as e:
        print(f"Error processing directory: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

