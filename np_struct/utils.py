import numpy as np

def check_shapes(a: tuple, b: tuple):
    """ 
    Check that the shape tuples a and b match
    """

    if len(a) != len(b):
        return False
    
    # check that the length of each dimension matches
    return all([a[i] == b[i] for i in range(len(a))])

def check_coords(c1: dict, c2: dict, tolerance=1e-6):
    """
    Check that two coordinates are identical
    """
    assert tuple(c1.keys()) == tuple(c2.keys()), f"Coord keys are different: {c1.keys()} vs {c2.keys()}"

    for k in c1.keys():
        if np.any(np.abs(c1[k] - c2[k]) > tolerance):
            return False

    return True