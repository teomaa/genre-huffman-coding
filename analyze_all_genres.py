# to generate a graph showing the average compression ratio per genre
# (the visual could be revised later)
import os
import subprocess
import re
import matplotlib.pyplot as plt

GENRES = [
    "ambient",
    "blues",
    "children",
    "classical",
    "country",
    "electronic",
    "folk",
    "jazz",
    "latin",
    "pop",
    "rap",
    "reggae",
    "religious",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
PKL_DIR = os.path.join(BASE_DIR, "genres")

MIDI_ROOT = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "adl-piano-midi",
    "adl-piano-midi",
)

VENV_PYTHON = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "huffman",
    ".venv",
    "Scripts",
    "python.exe",
)

avg_by_genre = {}
raw_logs_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(raw_logs_dir, exist_ok=True)

for genre in GENRES:
    print(f"PROCESSING GENRE: {genre.upper()}")

    midi_dir = os.path.join(MIDI_ROOT, genre.capitalize())
    pkl_path = os.path.join(PKL_DIR, f"{genre}.pkl")

    if not os.path.exists(midi_dir):
        print(f" MIDI folder missing: {midi_dir}")
        continue
    if not os.path.exists(pkl_path):
        print(f" PKL file missing: {pkl_path}")
        continue

    cmd = [
        VENV_PYTHON,
        "get_genre_huffman_code_v2.py",
        midi_dir,
        "--load_path",
        pkl_path,
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,      
        capture_output=True,
        text=True,
    )

   
    log_path = os.path.join(raw_logs_dir, f"{genre}_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)
        f.write("\n\n===== STDERR =====\n")
        f.write(result.stderr)


    match = re.search(r"Average compression ratio:\s*([0-9.]+)", result.stdout)
    if match:
        avg = float(match.group(1))
        avg_by_genre[genre] = avg
        print(f" {genre}: Average compression ratio = {avg:.2f}%")
    else:
        print(f" can not find average compression line for {genre}. Check {log_path}")


for g, v in sorted(avg_by_genre.items()):
    print(f"{g:10s} : {v:.2f}%")

if avg_by_genre:
    genres = list(avg_by_genre.keys())
    values = [avg_by_genre[g] for g in genres]

    plt.figure(figsize=(10, 5))
    plt.bar(genres, values)
    plt.xlabel("Genre")
    plt.ylabel("Average Compression Ratio (%)")
    plt.title("Average Huffman Compression Ratio per Genre")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("avg_compression_per_genre_from_stdout.png")
    plt.show()
else:
    print("bug  check logs")
