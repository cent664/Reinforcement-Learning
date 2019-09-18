import numpy

def inputlayerdim(W_out, F, P, S):

    W_in = ((W_out - 1) * S) - (2 * P) + F
    return W_in

def magic(params):
    # W = params[0]
    # F = params[1]
    # P = params[2]
    # S = params[3]
    
    for W_out in range(1, 100):
        W_in = inputlayerdim(W_out, F, P)

    list_of_acceptable_input_dims = []

    for i in range(W1, W1 + 50):
        W2 = outputlayerdim(i, F, P, S)
        if W2 % 1 == 0:
            list_of_acceptable_input_dims.append(i)
    if num_layers == 1:
        return list_of_acceptable_input_dims
    # else:
    #     return(magic(num_layers - 1, ))

if __name__ == '__main__':
    possible_dims = magic(1, [1, 8, 0, 4])
    print(possible_dims)