import numpy as np

def round_to_multiple(value, multiple=1):
    """ Rounds value to nearest multiple. Multiple can be greater or less than 1.

        Example:
            round_to_float(7.77777, 1e-3) --> 7.778

            round_to_float(7.77777, 3)    --> 9.0    
    """

    invmul = 1/multiple

    r1 = value/multiple
    w = r1//1
    w = np.where((r1%1) >= 0.5, w+1, w)
    
    mlog10 = np.log10(multiple)
    
    if mlog10 > 0:
        return w/invmul
    else:
        return np.around(w/invmul, int(np.abs(mlog10)))


def check_shapes(a, b):
    ## check that the shape length of a and b match
    if len(a) != len(b):
        return False
    
    ## check that the length of each dimension matches
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    
    return True