import numpy as np



#######################################################
# Utility functions
#######################################################
sec2microsec = lambda sec: int(np.round(sec * 1e+6))
microsec2sec = lambda microsec: microsec / 1e+6

def get_possible_tick_remainders(ticks_per_beat, notes_per_beat):
    """Get the remainders of ticks_per_beat when divided by notes_per_beat"""
    tick_remainders = [0, ticks_per_beat]
    for n in notes_per_beat:
        tick_remainders += [int(i * ticks_per_beat / n) for i in range(1, n)]
    # sort
    tick_remainders = sorted(set(tick_remainders))
    return tick_remainders

def round_tick_remainder(tick, tick_remainders, ticks_per_beat):
    """Round the tick to the nearest tick_remainder"""
    m = tick // ticks_per_beat
    r = tick % ticks_per_beat
    idx = np.argmin(np.abs(np.array(tick_remainders) - r))
    return int(m * ticks_per_beat + tick_remainders[idx])
