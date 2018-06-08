import re

NUM_CHARS = 30

with open('words_input.txt', 'r', encoding='ISO-8859-1') as fi_i, open('words_output.txt', 'r', encoding='ISO-8859-1') as fi_o, open('input_sentences.txt', 'w') as fo_i, open('output_sentences.txt', 'w') as fo_o:
    fi_i_fixed = fi_i.read().encode('ascii', 'ignore').decode('UTF-8').lower()
    fi_i_alpha = re.sub('[^a-z\n]+', ' ', fi_i_fixed)
    fi_o_fixed = fi_o.read().encode('ascii', 'ignore').decode('UTF-8').lower()
    fi_o_alpha = re.sub('[^a-z\n]+', ' ', fi_o_fixed)

    for line in fi_i_alpha.split('\n'):
        fo_i.write(line[:NUM_CHARS] + '\n')
    for line in fi_o_alpha.split('\n'):
        fo_o.write(line[:NUM_CHARS] + '\n')
