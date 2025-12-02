# genre-huffman-coding
Huffman coding for music genre classification

# How to run
- mkdir midi
- clone https://github.com/lucasnfe/adl-piano-midi/tree/master
- unzip /midi/adl-piano-midi.zip in that repo
- move all contents of /adl-piano-midi to /midi
- pip install -r requirements.txt
- To calculate the huffman code for one genre, do:
- python get_genre_huffman_code_v2.py '/Users/teo/genre-huffman-coding/midi/Classical' --save_path genres/classical.pkl # or any other genre directory, and any save path.
- you could also enter a sub-genre, for example '/Users/teo/genre-huffman-coding/midi/Classical/Ballroom' would work
- This saves a file, for example classical.pkl, to the genres/ directory. We can then use that file to compare against another genre. For example, i could compare how similar classical and jazz are, by doing:
- python get_genre_huffman_code_v2.py '/Users/teo/genre-huffman-coding/midi/Jazz' --load_path genres/classical.pkl
- Here, the output would be:
```
==================================================
COMPRESSION SUMMARY
==================================================
Highest compression: [Jazz] MFSB - Love Is The Message
  Compression ratio: 78.62%
  Chords: 1,344

Lowest compression: [Jazz] Ahmad Jamal Trio - Sweet And Lovely (2)
  Compression ratio: 0.00%
  Chords: 2,177

Average compression ratio: 24.16%
  (across 492 files)
```