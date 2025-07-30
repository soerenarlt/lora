import numpy as np
from itertools import combinations, product

from data_main.graphdata import *
from data_main.fancy_classes import Graph, State
import traceback


modes = ['bellN3d', 'ghz', 'w', 'dicke', 'dicke2dhalf2',
        'dicke2d2vsrest', 'dicke2d2vsrest2', 'dicke2d3vsrest2', 'ghzw', 'ww',
        'ghzghz', 'ghz3dghz3d', 'bellN', 'spinhalf', 'majumdar',
        'dyck', 'dyck246', 'aklt', 'motzkin', 'motzkinsmall']

def generate_state(ii, numvert, mode):
    print(f"Length of modes: {len(modes)}")
    if isinstance(mode, int):
        mode = modes[mode%len(modes)]
    if mode not in modes:
        raise ValueError(f"Mode {mode} not in {modes}")
    print(f"Mode: {mode}")

    if mode == 'ghz':    
        # #GHZ
        base_ket = [0]*numvert
        kets = []
        weights = None
        for ghz_mode in range(2):
            new_ket = base_ket.copy()
            new_ket = [ghz_mode]*(numvert)
            kets.append(new_ket)
        weights = None
        
    if mode == 'ghz3d2':
        # #GHZ
        kets = []
        weights = None
        for ghz_mode in range(3):
            new_ket = [ghz_mode]*(numvert-2)+[0]*2
            kets.append(new_ket)
        weights = None

    
    if mode == 'w':
        # #W MODE
        base_ket = [0]*numvert
        kets = []
        weights = None
        for w_pos in range(numvert):
            new_ket = base_ket.copy()
            new_ket[w_pos] = 1
            kets.append(new_ket)
        weights = None


    if mode == 'dicke':
        # #DICKE MODE (1,1,1), (2,1,1), (3,1,1)
        inds = list(combinations(range(ii), 2))
        print(2*len(inds))
        base_ket = [0]*numvert
        kets = []
        for ind in inds:
            new_ket = base_ket.copy()
            new_ket[ind[0]] = 1
            new_ket[ind[1]] = 2
            # new_ket=''.join(map(str, new_ket))
            kets.append(new_ket)

            new_ket = base_ket.copy()
            new_ket[ind[1]] = 1
            new_ket[ind[0]] = 2
            # new_ket=''.join(map(str, new_ket))
            kets.append(new_ket)
        weights = None

    # TOO MANY KETS (70 for 8 vertices)
    # if mode == 'dicke2d_half':
    #     # all combinations of half 0 half 1
    #     base_ket = [0]*numvert
    #     kets = []
    #     for ind in combinations(range(numvert), numvert//2):
    #         new_ket = base_ket.copy()
    #         for i in ind:
    #             new_ket[i] = 1
    #         kets.append(new_ket)

    if mode == 'dicke2dhalf2':
        # all combinations of half 0 half 1
        base_ket = [0]*numvert
        kets = []
        for ind in combinations(range(numvert-2), numvert//2 - 1):
            new_ket = base_ket.copy()
            for i in ind:
                new_ket[i] = 2
            kets.append(new_ket)
        weights = None

    if mode == 'dicke2d2vsrest':
        # all combinations of two 1s and the rest 0
        base_ket = [0]*numvert
        kets = []
        for ind in combinations(range(numvert), 2):
            new_ket = base_ket.copy()
            for i in ind:
                new_ket[i] = 1
            kets.append(new_ket)
        weights = None

    if mode == 'dicke2d2vsrest2':
        # all combinations of two 1s and the rest 0
        base_ket = [0]*numvert
        kets = []
        for ind in combinations(range(ii), 2):
            new_ket = base_ket.copy()
            for i in ind:
                new_ket[i] = 2
            kets.append(new_ket)
        weights = None

    # TOO MANY KETS
    # if mode == 'dicke2d_3vsrest':
    #     # all combinations of three 1s and the rest 0
    #     base_ket = [0]*numvert
    #     kets = []
    #     for ind in combinations(range(numvert), 3):
    #         new_ket = base_ket.copy()
    #         for i in ind:
    #             new_ket[i] = 1
    #         kets.append(new_ket)
            
    if mode == 'dicke2d3vsrest2':
        # all combinations of three 1s and the rest 0
        base_ket = [0]*numvert
        kets = []
        for ind in combinations(range(ii), 3):
            new_ket = base_ket.copy()
            for i in ind:
                new_ket[i] = 2
            kets.append(new_ket)
        weights = None


    if mode == 'ghzw':
        # #GHZ/W MODE
        base_ket = [0]*numvert
        kets = []
        for w_pos in range(numvert//2):
            for ghz_mode in range(2):
                new_ket = base_ket.copy()
                new_ket[w_pos+(numvert//2)] = 1
                new_ket[:numvert//2] = [ghz_mode]*(numvert//2)
                # new_ket=''.join(map(str, new_ket))
                kets.append(new_ket)
            # kets.append(new_ket)
        weights = None

    if mode == 'ww':
        # #W/W MODE
        base_ket = [0]*numvert
        kets = []
        for w_pos1 in range(numvert//2):
            for w_pos2 in range(numvert//2):
                new_ket = base_ket.copy()
                new_ket[w_pos1] = 1
                new_ket[w_pos2+(numvert//2)] = 1
                kets.append(new_ket)
        weights = None

    if mode == 'ghzghz':
        # #GHZ/GHZ MODE
        base_ket = [0]*numvert
        kets = []
        for ghz_mode1 in range(2):
            for ghz_mode2 in range(2):
                new_ket = base_ket.copy()
                new_ket[:numvert//2] = [ghz_mode1]*(numvert//2)
                new_ket[numvert//2:] = [ghz_mode2]*(numvert//2)
                kets.append(new_ket)
        weights = None

    if mode == 'ghz3dghz3d':
        base_ket = [0]*numvert
        kets = []
        for ghz_mode1 in range(3):
            for ghz_mode2 in range(3):
                new_ket = base_ket.copy()
                new_ket[:numvert//2] = [ghz_mode1]*(numvert//2)
                new_ket[numvert//2:] = [ghz_mode2]*(numvert//2)
                kets.append(new_ket)
        weights = None

    if mode == 'bellN':
        bell_terms = [[0,0],[1,1]]
        kets = []
        for comb in product(range(2), repeat=numvert//2):
            ket = []
            for c in comb:
                ket += bell_terms[c]
            kets.append(ket)
        weights = None

    if mode == 'bellN3d':
        bell_terms = [[0,0],[1,1],[2,2]]
        kets = []
        for comb in product(range(3), repeat=numvert//2-1):
            ket = []
            for c in comb:
                ket += bell_terms[c]
            ket += [0, 0]
            kets.append(ket)
        weights = None
    

    if mode == 'spinhalf':
    # spin 1/2 states
        base_ket = [0]*numvert
        kets = []
        #binaries from 0 to 2**numvert
        for jj in range(2**ii):
            new_ket = base_ket.copy()
            new_ket[:ii] = list(map(int, list(bin(jj)[2:].zfill(ii))))
            #check if [1, 1] is sublist of new_ket
            if [1, 1] in [new_ket[i:i+2] for i in range(len(new_ket)-1)]:
                continue
            else:
                kets.append(new_ket)
        print(len(kets))
        weights = None

    if mode == 'majumdar':
        A = [0]*2
        #Aup
        A[1] = np.matrix([[0, 1, 0], [0, 0, -1], [0, 0 , 0]])
        #Ado
        A[0] = np.matrix([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

        A_names = {0:'Au',1:'Ad'}

        # L should be even
        L = numvert
        # D should be 2
        D = 2
        base = [list(range(D))]*L
        sigmas = list(product(*base))
        kets = []
        weights = []
        for sigma in sigmas:
            mat = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
            for part in sigma:
                mat = np.matmul(mat,A[part])
            trace = round(np.trace(mat),3)
            if trace != 0:
                #print("*".join([A_names[ind] for ind in sigma]))
                # print(round(np.trace(mat),3))
                # print(sigma)
                # kets.append(''.join([str(el) for el in sigma]))
                #pad sigma to have length numvert
                sigma = list(sigma)
                sigma += [0]*(numvert - len(sigma))
                kets.append(sigma)
                weights.append(trace)

    if mode == 'dyck':
        #dyck words mode
        #all possible dyck words of length numvert
        # dyck words mode
        kets = []
        def generate_dyck_words(word, open_count, close_count):
            if open_count == 0 and close_count == 0:
                kets.append(word)
                return
            if open_count > 0:
                generate_dyck_words(word + [1], open_count - 1, close_count)
            if close_count > open_count:
                generate_dyck_words(word + [2], open_count, close_count - 1)

        generate_dyck_words([], numvert // 2, numvert // 2)
        weights = None

    if mode == 'dyck246':
        #dyck words mode
        #all possible dyck words of length numvert
        # dyck words mode
        kets = []
        def generate_dyck_words(word, open_count, close_count):
            if open_count == 0 and close_count == 0:
                kets.append(word+[0,0])
                return
            if open_count > 0:
                generate_dyck_words(word + [1], open_count - 1, close_count)
            if close_count > open_count:
                generate_dyck_words(word + [2], open_count, close_count - 1)

        generate_dyck_words([], numvert // 2 - 1, numvert // 2 - 1)
        weights = None

    if mode == 'aklt':

        A = [0]*3
        #A+
        A[0] = np.matrix([[0, 1/np.sqrt(2)], [0, 0]])
        #A0
        A[1] = np.matrix([[-1/2, 0], [0, 1/2]])
        #A-
        A[2] = np.matrix([[0, 0], [-1/np.sqrt(2), 0]])

        A_names = {0:'A+',1:'A0', 2:'A-'}

        L = ii - 1
        D = 3
        base = [list(range(D))]*L
        sigmas = list(product(*base))
        kets = []
        weights = []
        for sigma in sigmas:
            mat = np.matrix([[1,0],[0,1]])
            for part in sigma:
                mat = np.matmul(mat,A[part])
            trace = round(np.trace(mat),12)
            if trace != 0:
                #print("*".join([A_names[ind] for ind in sigma]))
                # print(round(np.trace(mat),12))
                # print(sigma)
                # kets.append(''.join([str(el) for el in sigma]))
                #pad sigma to have length numvert
                sigma = list(sigma)
                sigma += [0]*(numvert - len(sigma))

                kets.append(sigma)
                weights.append(int(trace*(2**(L-1)))) #MULTIPLY BY L TO GET int COEFFICIENTS

    if mode == 'motzkin':
        motzkin_symbols = ['(', ')', '-']

        def is_motzkin_word(word):
            depth = 0
            for symbol in word:
                if motzkin_symbols[symbol] == '(':
                    depth += 1
                elif motzkin_symbols[symbol] == ')':
                    depth -= 1
                    if depth < 0:
                        return False
            return depth == 0

        def generate_motzkin_words(N):
            canditates = list(product([0,1,2], repeat=N))
            words = []
            for candidate in canditates:
                if is_motzkin_word(candidate):
                    words.append(candidate)
            print(len(words))
            return words

        motzkin_words = generate_motzkin_words(ii)
        base_ket = [0]*numvert
        kets = []
        for motzkin_word in motzkin_words:
            new_ket = base_ket.copy()
            new_ket[:ii] = motzkin_word
            kets.append(new_ket)
        weights = None

    if mode == 'motzkinsmall':
        motzkin_symbols = ['(', ')', '-']

        def is_motzkin_word(word):
            depth = 0
            for symbol in word:
                if motzkin_symbols[symbol] == '(':
                    depth += 1
                elif motzkin_symbols[symbol] == ')':
                    depth -= 1
                    if depth < 0:
                        return False
            return depth == 0

        def generate_motzkin_words(N):
            canditates = list(product([0,1,2], repeat=N))
            words = []
            for candidate in canditates:
                if is_motzkin_word(candidate):
                    words.append(candidate)
            print(len(words))
            return words

        motzkin_words = generate_motzkin_words(ii-1)
        base_ket = [0]*numvert
        kets = []
        for motzkin_word in motzkin_words:
            new_ket = base_ket.copy()
            new_ket[:ii-1] = motzkin_word
            kets.append(new_ket)
        weights = None


    if weights is None:
        weights = np.ones(len(kets), dtype=np.int64)
    zipped = list(zip(kets, weights))
    #sort by kets
    zipped.sort(key=lambda x: x[0])
    kets, weights = zip(*zipped)
    
    state = State({tuple([(pos, dim) for pos, dim in enumerate(ket)]):weights[ii] for ii, ket in enumerate(kets)})

    # print(kets)
    if weights is not None:
        state = State({tuple([(pos, dim) for pos, dim in enumerate(ket)]):weights[ii] for ii, ket in enumerate(kets)})
    else:
        state = State({tuple([(pos, dim) for pos, dim in enumerate(ket)]):1 for ket in kets}, normalize=False)

    return state