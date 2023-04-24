from fontcompare import diff_many_words

print("Reading wordlist...")
wordlist = []
with open("test-data/Arabic.txt") as f:
  for line in f:
    wordlist.append(line.rstrip())

print("Diffing...")
output = diff_many_words(
            "test-data/NotoSansArabic-Old.ttf",
            "test-data/NotoSansArabic-New.ttf",
            20.0,
            wordlist,
            10.0)

for word, buffer_a, buffer_b, percent in output:
  print(word, percent)
