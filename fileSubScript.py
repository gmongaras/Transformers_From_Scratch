# This script cuts an input file down so that the number
# of words per line does not pass a certain threshold


# The max sentence length we want
maxLen = 100

# The max number of lines we want
maxLines = 2000000

# Read in the input files
file1_I = open("data/english.txt", mode="r", encoding="utf-8")
file2_I = open("data/spanish.txt", mode="r", encoding="utf-8")

# Open the output files
file1_O = open("data/english_sub.txt", mode="w", encoding="utf-8")
file2_O = open("data/spanish_sub.txt", mode="w", encoding="utf-8")



# Get all lines from the input files
lines1 = file1_I.readlines()
lines2 = file2_I.readlines()


# Iterate over all lines in each of the files
lineCt = 0
for i in range(0, len(lines1)):
    # If the number of spaces in each line is less than the specified length,
    # add the lines to the files
    if (lines1[i].count(" ") < maxLen and lines2[i].count(" ") < maxLen):
        file1_O.write(lines1[i])
        file2_O.write(lines2[i])
        lineCt += 1
    
    # If the line count is larger than the line count wanted, break the loop
    if (lineCt >= maxLines and maxLines != -1):
        break


# Close the files
file1_I.close()
file2_I.close()
file1_O.close()
file2_O.close()




file = "data/spanish_sub.txt"
f = open(file, mode='r', encoding="utf-8")
lines = f.readlines()
longest = ""
for l in lines:
    if len(l) > len(longest):
        longest = l
print(f"Longest word: {longest}")
print(f"Number of lines: {len(lines)}")
f.close()
