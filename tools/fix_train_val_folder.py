import os
with open('/disk4/something-something-v2/train_videofolder_new.txt', 'w') as outfile:
    with open('/disk4/something-something-v2/train_videofolder.txt') as infile:
        for line in infile.readlines():
            newline = line.split(' ')[0] + '.webm ' + line.split(' ')[-1]
            outfile.write(newline)

with open('/disk4/something-something-v2/val_videofolder_new.txt', 'w') as outfile:
    with open('/disk4/something-something-v2/val_videofolder.txt') as infile:
        for line in infile.readlines():
            newline = line.split(' ')[0] + '.webm ' + line.split(' ')[-1]
            outfile.write(newline)