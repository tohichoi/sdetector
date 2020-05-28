from pomegranate import *
import numpy as np


def modelparam1():
    dists = [
        DiscreteDistribution({'0': 0.9, '1': 0.1}),
        DiscreteDistribution({'0': 0.9, '1': 0.1}),
        DiscreteDistribution({'0': 0.9, '1': 0.1}),
        DiscreteDistribution({'0': 0.8, '1': 0.2}),
        DiscreteDistribution({'0': 0.9, '1': 0.1}),
    ]
    transmat = np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                         [0.0, 0.8, 0.2, 0.0, 0.0],
                         [0.0, 0.1, 0.0, 0.9, 0.0],
                         [0.0, 0.0, 0.0, 0.8, 0.2],
                         [0.0, 0.9, 0.1, 0.0, 0.0]
                         ])
    startprob = np.array([0.1, 0.4, 0.1, 0.4, 0.1])
    endprob = np.array([0.0, 0.4, 0.1, 0.4, 0.1])
    statename = 'UAEPL'

    return dists, transmat, startprob, endprob, statename


def modelparam2():
    dists = [
        DiscreteDistribution({'0': 0.7, '1': 0.3}),
        DiscreteDistribution({'0': 0.7, '1': 0.3}),
    ]
    transmat = np.array([
        [0.8, 2.0],
        [0.5, 0.5]
    ])
    startprob = np.array([0.5, 0.5])
    endprob = np.array([0.5, 0.5])
    statename = 'AP'

    return dists, transmat, startprob, endprob, statename


dists, transmat, startprob, endprob, statename = modelparam1()

model = HiddenMarkovModel.from_matrix(transmat, dists, startprob, endprob)

model.bake()

with open('hmm.json', 'w+') as fd:
    fd.write(model.to_json())

with open('hmmtest-act3.txt') as fd:
    seqs = fd.readline().strip()
    nwin = 2
    for i in range(len(seqs) - nwin):
        # seq=np.array(list(map(lambda x : int(x), seqs[i:i+nwin])))
        nwin = min(nwin + 1, 40)
        seq = np.array(list(seqs[i:i + nwin]))
        pred = model.predict(seq)

        print("sequence: {}".format(''.join(seq)))
        print(f"hmm pred: {''.join(list(map(lambda x: statename[x], pred)))}")
        print(f"log prob: {model.log_probability(seq):.2f}")
