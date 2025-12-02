#!/usr/bin/env python3
"""
Script to build a genre-wide Huffman code from all MIDI files in a directory
and analyze compression ratios for individual songs.

Symbols are chord representations (pitch-class sets) sampled on a time grid.
This captures harmonic content and is transposition-invariant.
Chords are computed within each song (reset after each song).
"""

import argparse
import sys
import pickle
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


def build_huffman_tree(chords: List) -> Tuple[HuffmanNode, Dict]:
    """Build Huffman tree and return root node and encoding dictionary"""
    # Count frequencies of chords
    frequencies = Counter(chords)
    
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


def encode_data(chords: List, encoding_dict: Dict) -> str:
    """Encode chord data using Huffman encoding dictionary"""
    encoded_bits = ''.join(encoding_dict[chord] for chord in chords)
    return encoded_bits


def decode_data(encoded_bits: str, root: HuffmanNode, expected_length: int) -> List:
    """Decode Huffman-encoded bits back to original chord data"""
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
    
    return decoded_values


def calculate_compressed_size(encoded_bits: str, encoding_dict: Dict) -> int:
    """Calculate size of compressed data in bits"""
    # Size of encoded data
    data_size = len(encoded_bits)
    
    # Size of encoding dictionary (approximate)
    # For each entry: chord representation + code length (1 byte) + code bits
    dict_size = 0
    for chord, code in encoding_dict.items():
        # Chord is a frozenset, estimate size: number of pitch classes (1 byte) + pitch classes (1 byte each)
        if isinstance(chord, frozenset):
            dict_size += 1  # number of pitch classes
            dict_size += len(chord)  # pitch class values
        else:
            dict_size += 4  # fallback for other types
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


def extract_all_genre_chords(directory: str, time_resolution: float = 0.125) -> Tuple[List, List[Tuple[str, str, str, str, List]]]:
    """
    Extract chords from all MIDI files in directory.
    Returns: (all_chords, list of (file_path, genre, artist, song, chords) for each file)
    Chords are computed within each song (reset after each song).
    
    Args:
        directory: Directory containing MIDI files
        time_resolution: Time step in seconds (default 0.125 = 16th note at 120 BPM)
    """
    midi_files = find_midi_files(directory)
    
    if len(midi_files) == 0:
        raise ValueError(f"No MIDI files found in directory: {directory}")
    
    all_chords = []
    file_chords = []
    
    print(f"Found {len(midi_files)} MIDI files")
    print(f"Extracting chords with time resolution: {time_resolution}s (16th note grid)...")
    
    # Use tqdm for progress bar
    for file_path, genre, artist, song in tqdm(midi_files, desc="Processing MIDI files", unit="file"):
        try:
            midi_data = pretty_midi.PrettyMIDI(file_path)
            chords = extract_chords(midi_data, time_resolution)
            
            if len(chords) > 0:
                all_chords.extend(chords)
                file_chords.append((file_path, genre, artist, song, chords))
        except Exception:  # noqa: BLE001
            # Continue processing other files if one fails
            # Catch all exceptions including MIDI parsing errors (KeySignatureError, etc.)
            # from mido/pretty_midi libraries, which may raise various exception types
            continue
    
    if len(all_chords) == 0:
        raise ValueError("No chords could be extracted from any MIDI files")
    
    return all_chords, file_chords


def extract_chords(midi_data: pretty_midi.PrettyMIDI, time_resolution: float = 0.125) -> List[frozenset]:
    """
    Extract chords from MIDI file using a time grid.
    
    Args:
        midi_data: PrettyMIDI object
        time_resolution: Time step in seconds (default 0.125 = 16th note at 120 BPM)
    
    Returns:
        List of chord representations (frozensets of pitch classes 0-11)
    """
    # Get the total duration of the MIDI file
    total_time = midi_data.get_end_time()
    
    if total_time == 0:
        raise ValueError("MIDI file has zero duration")
    
    # Create time grid
    time_points = np.arange(0, total_time, time_resolution)
    
    chords = []
    
    for t in time_points:
        # Get all active pitches at this time point
        active_pitches = []
        
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                # Check if note is active at time t
                if note.start <= t < note.end:
                    active_pitches.append(note.pitch)
        
        # Convert pitches to pitch classes (mod 12) for transposition invariance
        pitch_classes = frozenset(pitch % 12 for pitch in active_pitches)
        
        # Store the chord (empty set represents silence/rest)
        chords.append(pitch_classes)
    
    if len(chords) == 0:
        raise ValueError("No chords could be extracted from MIDI file")
    
    return chords


def print_huffman_code(encoding_dict: Dict):
    """Print the Huffman code in a readable format"""
    print("\n" + "="*50)
    print("GENRE HUFFMAN CODE")
    print("="*50)
    
    # Sort by code length, then by chord representation
    sorted_codes = sorted(encoding_dict.items(), key=lambda x: (len(x[1]), sorted(x[0]) if isinstance(x[0], frozenset) else x[0]))
    
    print(f"{'Chord (Pitch Classes)':<25} {'Code':<20} {'Length'}")
    print("-" * 70)
    for chord, code in sorted_codes:
        if isinstance(chord, frozenset):
            if len(chord) == 0:
                chord_str = "{} (rest)"
            else:
                chord_str = "{" + ",".join(map(str, sorted(chord))) + "}"
        else:
            chord_str = str(chord)
        print(f"{chord_str:<25} {code:<20} {len(code)}")
    
    print("="*50)


def save_huffman_code(encoding_dict: Dict, root: HuffmanNode, file_path: str):
    """Save Huffman code to disk using pickle"""
    data = {
        'encoding_dict': encoding_dict,
        'root': root
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Huffman code saved to: {file_path}")


def load_huffman_code(file_path: str) -> Tuple[Dict, HuffmanNode]:
    """Load Huffman code from disk using pickle"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Huffman code loaded from: {file_path}")
        return data['encoding_dict'], data['root']
    except FileNotFoundError:
        print(f"Error: Huffman code file not found: {file_path}")
        sys.exit(1)
    except (pickle.UnpicklingError, KeyError) as e:
        print(f"Error: Invalid or corrupted Huffman code file: {e}")
        sys.exit(1)


def compress_file_with_code(chords: List, encoding_dict: Dict, genre: str, artist: str, song: str) -> float:
    """
    Compress a single file's chords using the provided Huffman code.
    Returns compression ratio as a percentage.
    Note: Dictionary size is not included since it's shared across all files.
    """
    if len(chords) == 0:
        return 0.0
    
    # Check if all chords in this file are in the encoding dictionary
    missing_chords = set(chords) - set(encoding_dict.keys())
    if missing_chords:
        # This shouldn't happen if we built the code from all files, but handle it
        print(f"  Warning: {len(missing_chords)} chord types not in code for [{genre}] {artist} - {song}")
        return 0.0
    
    # Encode using the genre code
    encoded_bits = encode_data(chords, encoding_dict)
    
    # Calculate sizes
    # Original: estimate size of chord data
    # Each chord is a frozenset, estimate: 1 byte per pitch class + overhead
    original_size = sum(1 + len(chord) for chord in chords)  # Rough estimate
    
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
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Path to save the Huffman code (pickle file)'
    )
    parser.add_argument(
        '--load_path',
        type=str,
        default=None,
        help='Path to load a Huffman code from (pickle file). If provided, code computation is skipped.'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.save_path and args.load_path:
        print("Error: Cannot specify both --save_path and --load_path")
        sys.exit(1)
    
    try:
        # Load or build Huffman code
        if args.load_path:
            # Load existing code
            print(f"Loading Huffman code from: {args.load_path}")
            encoding_dict, root = load_huffman_code(args.load_path)
            
            if not encoding_dict or root is None:
                print("Error: Invalid Huffman code file")
                sys.exit(1)
            
            print(f"Loaded code with {len(encoding_dict)} unique chord types")
            
            # Still need to extract chords from directory for compression testing
            print(f"\nProcessing directory: {args.directory}")
            _, file_chords = extract_all_genre_chords(args.directory)
            
        else:
            # Build new code
            # Extract chords from all files in directory
            print(f"Processing directory: {args.directory}")
            all_chords, file_chords = extract_all_genre_chords(args.directory)
            
            print(f"\nTotal chords across all files: {len(all_chords):,}")
            unique_chords = len(set(all_chords))
            print(f"Number of unique chord types: {unique_chords:,}")
            
            # Build genre-wide Huffman tree
            print("\nBuilding genre-wide Huffman tree...")
            root, encoding_dict = build_huffman_tree(all_chords)
            
            if not encoding_dict or root is None:
                print("Error: Could not build encoding dictionary")
                sys.exit(1)
            
            print(f"Number of unique chord types in code: {len(encoding_dict)}")
            
            # Save code if requested
            if args.save_path:
                save_huffman_code(encoding_dict, root, args.save_path)
        
        # Print the Huffman code (commented out by user, but available)
        # print_huffman_code(encoding_dict)
        
        # Compress each individual file using the genre code
        print("\n" + "="*50)
        print("COMPRESSING INDIVIDUAL FILES")
        print("="*50)
        
        compression_results = []
        
        for _file_path, genre, artist, song, chords in tqdm(file_chords, desc="Compressing files", unit="file"):
            compression_ratio = compress_file_with_code(chords, encoding_dict, genre, artist, song)
            compression_results.append((genre, artist, song, compression_ratio, len(chords)))
        
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
        print(f"  Chords: {highest[4]:,}")
        print()
        print(f"Lowest compression: [{lowest[0]}] {lowest[1]} - {lowest[2]}")
        print(f"  Compression ratio: {lowest[3]:.2f}%")
        print(f"  Chords: {lowest[4]:,}")
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

