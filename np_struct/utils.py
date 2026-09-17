import numpy as np

DATA_FMT_FUNC = {
    "mag": np.abs,
    "db20": lambda x: 20 * np.log10(np.abs(x)),
    "db10": lambda x: 10 * np.log10(np.abs(x)),
    "deg": lambda x: np.angle(x, deg=True),
    "rad": lambda x: np.angle(x, deg=False),
    "angle": lambda x: np.angle(x, deg=False),
    "deg_unwrap": lambda x: np.rad2deg(np.unwrap(np.angle(x))),
    "real": np.real,
    "imag": np.imag,
}

LABEL_FMT_FUNC = dict(
    frequency = lambda x: f"{x/1e9:.3f}GHz"
)


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

def add_data_formatter(key: str, func):
    """
    Add a custom data format function. Function must accept a single numpy array argument and return a 
    formatted array of the same shape.
    """
    DATA_FMT_FUNC[key] = func

def add_label_formatter(key: str, func):
    """
    Add a custom data format function. Function must accept a single scalar argument and return a 
    formatted string
    """
    LABEL_FMT_FUNC[key] = func

def format_label(
    key: str, value
) -> str:

    if key in LABEL_FMT_FUNC.keys():
        return LABEL_FMT_FUNC[key](value)

    # create default label formatters if not included in look up table
    if isinstance(value, (float, np.floating)):
        return f"{key}={value:.3f}"
    elif isinstance(value, (int, np.integer)):
        return f"{key}={value}"
    # don't include key in label for string coordinates
    else:
        return "{}".format(value)
