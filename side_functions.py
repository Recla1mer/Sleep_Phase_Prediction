"""
Author: Johannes Peter Knoll

In this file we provide functions that are used to keep the code a little bit cleaner.
"""

# IMPORTS
import time
import numpy as np


"""
=============================
Printing Progress to console
=============================
"""


def print_smart_float(floating_point_number: float, decimals: int) -> str:
    """
    Convert a floating point number to a string with a certain number of decimals.

    ARGUMENTS:
    ------------------------------
    floating_point_number: float
        the number to convert
    decimals: int
        the number of decimals to keep
    
    RETURNS:
    ------------------------------
    str
        the number as a string with the specified number of decimals
    """

    count_division_by_10 = 0
    while True:
        if floating_point_number < 10:
            break
        else:
            floating_point_number /= 10
            count_division_by_10 += 1
    
    count_multiply_by_10 = 0
    if count_division_by_10 == 0:
        while True:
            if floating_point_number >= 1:
                break
            else:
                floating_point_number *= 10
                count_multiply_by_10 += 1
    
    floating_point_number = round(floating_point_number, decimals)

    if count_division_by_10 > 0:
        return str(floating_point_number) + " e" + str(count_division_by_10)
    elif count_multiply_by_10 > 0:
        return str(floating_point_number) + " e-" + str(count_multiply_by_10)
    else:
        return str(floating_point_number)


def print_smart_time(time_seconds: float) -> str:
    """
    Convert seconds to a time format that is easier to read.

    ARGUMENTS:
    ------------------------------
    time_seconds: int
        time in seconds
    
    RETURNS:
    ------------------------------
    str
        time in a more readable format
    """

    if time_seconds <= 1:
        return str(round(time_seconds, 1)) + "s"
    else:
        time_seconds = round(time_seconds)
        days = time_seconds // 86400
        if days > 0:
            time_seconds = time_seconds % 86400
        hours = time_seconds // 3600
        if hours > 0:
            time_seconds = time_seconds % 3600
        minutes = time_seconds // 60
        seconds = time_seconds % 60

        if days > 0:
            return str(days) + "d " + str(hours) + "h"
        if hours > 0:
            return str(hours) + "h " + str(minutes) + "m"
        elif minutes > 0:
            return str(minutes) + "m " + str(seconds) + "s"
        else:
            return str(seconds) + "s"


def progress_bar(index: int, total: int, batch_size: int, start_time: float, loss = None, corr_prop = None, decimals = 3, bar_len=40, max_length = 130):
    """
    Prints a progress bar to the console.

    Idea taken from:
    https://stackoverflow.com/questions/6169217/replace-console-output-in-python

    ARGUMENTS:
    ------------------------------
    index: int
        current index
    total: int
        total number
    bar_len: int
        length of the progress bar
    title: str
        title of the progress bar

    RETURNS:
    ------------------------------
    None, but prints the progress bar to the console
    """

    if total == 0:
        return

    # estimate time remaining
    if index == 0:
        time_passed = None
    else:
        time_passed = time.time() - start_time
        time_per_index = time_passed/index
        time_total = time_per_index*total
        time_remaining = time_per_index*(total-index)
        time_per_index *= batch_size

        time_passed_str = print_smart_time(time_passed)
        time_per_index_str = print_smart_time(time_per_index)
        time_remaining_str = print_smart_time(time_remaining)
        time_total_str = print_smart_time(time_total)

    # code from source
    percent_done = index/total*100
    rounded_percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    if time_passed is None:
        message = f'   ⏳: {rounded_percent_done}% [{done_str}{togo_str}] {index} / {total}'
    elif loss is None or corr_prop is None:
        message = f'   ⏳: {rounded_percent_done}% [{done_str}{togo_str}] {index} / {total} | {time_passed_str} / {time_total_str} ({time_per_index_str}/it)'
    else:
        message = f'   ⏳: {rounded_percent_done}% [{done_str}{togo_str}] {index} / {total} | {time_passed_str} / {time_total_str} ({time_per_index_str}/it) | Loss: {print_smart_float(loss, decimals)} | Acc: {round(corr_prop*100, 2)}%'

    print(message + " "*(max_length-len(message)), end='\r')

    if index >= total:
        print('   ✅')