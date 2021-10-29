import argparse
import pickle

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--data-in", help="Path to glove file", default="data/glove.6B.200d.txt")
    args.add_argument("--data-out", help="Path to output glove dictionary", default="data/glove.pkl")
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()

    glove = {}
    with open(args.data_in, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            embd = [float(x) for x in line[1:]]
            glove[word] = embd

    # random checks
    assert "the" in glove
    assert "The" not in glove
    assert "and" in glove
    assert "And" not in glove

    with open(args.data_out, "wb") as f:
        pickle.dump(glove, f)