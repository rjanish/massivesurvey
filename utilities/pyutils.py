""" Miscellaneous useful functions for programming tasks """


import re
import collections
import copy

import numpy as np


re_decimal = r"[-+]?(?:\d+\.?\d*|\.\d+)"
    # This regular expression pattern matches decimal numbers in the form
    # '13', '1.3', '13.', and '.13', and with an optional leading '+' or '-'.
re_yesno = r"[Yy]|(YES|Yes|yes)|[Nn]|(NO|No|no)"
    # matches common text variations of 'yes' and 'no'


def force_full_match(re_pattern):
    """
    Returns a new re pattern that will match anything that is also
    matched by the passed pattern, with the added criteria that the
    match must now cover the entire string under comparison.
    """
    if re_pattern[0] != '^':
        re_pattern = re_pattern[1:]
    if re_pattern[-1] != '$':
        re_pattern = re_pattern[:-1]
    return r"^({})$".format(re_pattern)


def match_bool(pattern, string):
    """ Does pattern match the entirety of string? Returns boolean. """
    full_pattern = force_full_match(pattern)
    match = re.match(full_pattern, string)
    return bool(match)


def validated_input(prompt, validator=None, fail_action='repeat'):
    """
    Gather command line input with validation.

    Args:
    prompt - string
        The prompt to be printed to the user
    validator - function, accepts one string and returns one bool
        This the validation function - acting on a string, it should
        return True for a valid input and False for an invalid input.
        By default, it accepts all strings a valid.
    fail_action - one of the following keywords, default is 'repeat':
        'repeat' - continually prompt user until valid input is given
        'exception' - raise ValueError on invalid input
        'none' - silently return None on invalid input

    Returns: response
    response - string
        The valid input or None, depending on value of fail_action.
    """
    if validator is None:
        validator = lambda s: True  # all strings are valid
    valid = False  # always prompt at least once
    repeat = (fail_action == 'repeat')
    while not valid:
        response = raw_input(prompt)
        valid = validator(response)
        if not repeat:
            break
        # loop exits if and only if response is valid or mode is not 'repeat'
    if valid:
        return response
    elif fail_action == 'none':
        return None
    elif fail_action == 'exception':
        raise ValueError("Invalid Response: {}".format(response))


def yesno_to_bool(string, whine=False):
    """
    Convert string variants of 'Yes' and "No" to boolean.

    On failure, this will raise a value error unless 'whine' is
    set to False, in which case it silently returns None.
    """
    cleaned_string = string.strip().lower()
    valid_yesno = re.match(force_full_match(re_yesno), cleaned_string)
    if not valid_yesno:
        if whine:
            raise ValueError("Invalid input: {} cannot be interpreted as a "
                             "'yes' or 'no' string.".format(string))
        else:
            return None
    starting_char = cleaned_string[0]  # must be 'y' or 'n', due to re_yesno
    if starting_char == 'y':
        output = True
    elif starting_char == 'n':
        output = False
    return output

    
def safe_int(value, fail_value=np.nan):
    """
    Converts to integer type if possible, else returns fail_value.
    """
    try:
        return int(value)
    except:
        return fail_value


def safe_str(value, fail_value='--'):
    """
    Converts to non-empty string type, if possible. The fail_value is
    returned if value cannot be cast to string or is an empty string.
    """
    try:
        to_string = str(value)
        if to_string:
            return to_string
        else:
            return fail_value
    except:
        return fail_value


def safe_float(value, fail_value=np.nan):
    """
    Converts to float type if possible, else returns fail_value.
    """
    try:
        return float(value)
    except:
        return fail_value


def append_to_dict(dest, *dicts):
    """
    Returns a key-wise appended dictionary. All values in the passed
    dicts are appended to the corresponding values in dest, matched by
    key. The values are appended in the order of the passed dicts. If
    dest does not contain a key that is present in some of the dicts,
    then the output is a list of all such values in dicts.
    """
    output = collections.defaultdict(list)
    output.update(copy.deepcopy(dest))
    for input_dict in dicts:
        for key, value in input_dict.iteritems():
            output[key].append(value)
    return dict(output)
    

def merge_dicts(*dicts):
    """
    Returns a single merged dictionary. This dictionary has one key
    for each unique key appearing in any of the passed dictionaries,
    with the value being a list of all of the corresponding values
    from the passed dictionaries.  The order of values in the list
    will match the order of the passed dictionaries.
    """
    return append_to_dict({}, *dicts)
