import os
import csv
import random
from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm

# I'm using the chord-based helpers from the existing file
from get_genre_huffman_code_v2 import (
    extract_all_genre_chords,   
    build_huffman_tree,
    compress_file_with_code,
)

#I start with a few genres so it runs faster (CAN CHANGE IT LATER) 
GENRES = ["jazz", "pop", "classical"]  

# Limit number of songs per genre (CAN CHANGE IT)
MAX_SONGS_PER_GENRE = 30

# my MIDI root:  ...\Downloads\adl-piano-midi\adl-piano-midi\<Genre>\
MIDI_ROOT = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "adl-piano-midi",
    "adl-piano-midi",
)

OUTPUT_CSV = "cross_genre_small_chords_results.csv"


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # genre -> encoding_dict
    genre_codes: Dict[str, Dict[int, str]] = {}
    all_files: List[Dict[str, Any]] = []

    print("=== BUILDING HUFFMAN CODES (CHORD-BASED) FOR SELECTED GENRES ===")

    for genre in GENRES:
        midi_dir = os.path.join(MIDI_ROOT, genre.capitalize())
        if not os.path.isdir(midi_dir):
            print(f"Skipping {genre}: no MIDI folder at {midi_dir}")
            continue

        print(f"\nProcessing genre folder: {midi_dir}")
        all_chords, file_chords = extract_all_genre_chords(midi_dir)

        if len(file_chords) == 0:
            print(f"No valid files found for {genre}, skipping.")
            continue

        # Randomly sample files
        if len(file_chords) > MAX_SONGS_PER_GENRE:
            file_chords = random.sample(file_chords, MAX_SONGS_PER_GENRE)
            print(f"  Using a random sample of {MAX_SONGS_PER_GENRE} songs for {genre}.")

      
        sampled_chord_arrays = [fc[4] for fc in file_chords]  # index 4 is chords array per file
        sampled_all_chords = np.concatenate(sampled_chord_arrays)
        _, encoding_dict = build_huffman_tree(sampled_all_chords)
        genre_codes[genre] = encoding_dict
        print(f"  Built Huffman code for {genre} with {len(encoding_dict)} unique chord types")

        # Store song data 
        for file_path, file_genre, artist, song, chords in file_chords:
            all_files.append(
                {
                    "file_path": file_path,
                    "artist": artist,
                    "song": song,
                    "true_genre": file_genre.lower(),
                    "chords": chords,
                }
            )

    if not genre_codes:
        print("No genre codes were built. Check folder paths / genres.")
        return

    print(f"\nCollected {len(all_files)} songs total across selected genres.")
    print("=== CROSS-GENRE COMPRESSION (CHORDS) ===")

    results: List[Dict[str, Any]] = []

    for song_info in tqdm(all_files, desc="Cross-compressing songs"):
        chords = song_info["chords"]
        true_genre = song_info["true_genre"]

        per_genre_ratios: Dict[str, float] = {}
        best_genre = None
        best_ratio = None
        own_ratio = None

        for genre_name, encoding_dict in genre_codes.items():
           
            ratio = compress_file_with_code(
                chords,
                encoding_dict,
                genre_name,
                song_info["artist"],
                song_info["song"],
            )

            
            # if our implementation returns 0.0 for "missing chords", that's okay; those cases just won't win
            per_genre_ratios[genre_name] = ratio

            if genre_name == true_genre:
                own_ratio = ratio

            if (best_ratio is None) or (ratio > best_ratio):
                best_ratio = ratio
                best_genre = genre_name

        if best_genre is None:
            continue

        row = {
            "file_path": song_info["file_path"],
            "artist": song_info["artist"],
            "song": song_info["song"],
            "true_genre": true_genre,
            "best_genre": best_genre,
            "best_ratio": best_ratio,
            "own_ratio": own_ratio,
        }

        # store per-genre ratios
        for g in GENRES:
            row[f"ratio_{g}"] = per_genre_ratios.get(g, None)

        results.append(row)

    if not results:
        print("No cross-genre results produced.")
        return

    out_path = os.path.join(base_dir, OUTPUT_CSV)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n Saved small cross-genre chord-based results to: {out_path}")

    # test if there's circumstances where best_genre ≠ true_genre
    anomalies = [r for r in results if r["best_genre"] != r["true_genre"]]
    print(f"\nFound {len(anomalies)} songs where best_genre ≠ true_genre.")
    for r in anomalies[:10]:
        print(
            f"- {r['song']} (true: {r['true_genre']}, best: {r['best_genre']}, "
            f"own_ratio={r['own_ratio']}, best_ratio={r['best_ratio']})"
        )


if __name__ == "__main__":
    main()
