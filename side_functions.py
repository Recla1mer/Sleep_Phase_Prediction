"""
Author: Johannes Peter Knoll

In this file we provide functions that are used to keep the code a little bit cleaner.
"""

# IMPORTS
import time
import numpy as np
import os
import shutil
import sys
import tty
import termios
import subprocess
import platform
import signal

from IPython import get_ipython # type: ignore
from IPython.display import clear_output


"""
========================
Printing Console Output
========================
"""

def print_headline(headline: str, symbol_sequence: str = "="):
    """
    Print a headline highlighted with a symbol.

    ARGUMENTS:
    ------------------------------
    headline: str
        the headline to print
    symbol_sequence: str
        the symbol or symbol sequence to box the headline with
    """

    width = len(headline)
    repeat_symbol = int(np.ceil(width / len(symbol_sequence)))

    print("\n\n" + symbol_sequence*repeat_symbol)
    print(headline)
    print(symbol_sequence*repeat_symbol)


"""
=============
Progress Bar
=============
"""


def format_float(floating_point_number: float, decimals: int) -> str:
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


def format_time(seconds: float, max_parts: int = 2, written_unit: bool = False) -> str:
    """
    Converts a time duration in seconds into a human-readable format. 
    
    If the time duration is larger than 60 seconds, the time will be displayed using a combination of
    units like days, hours, minutes, and seconds. 
    
    If the time duration is smaller than 60 seconds, the time will be displayed using SI prefixes like
    milliseconds, microseconds, and nanoseconds.

    ARGUMENTS:
    ------------------------------
    seconds: float
        time duration in seconds
    max_parts: int
        maximum number of time parts to display for large time durations (e.g., 1d 2h 3m 4s = 4 parts)
    written_unit: bool
        whether to write out the unit names (e.g., "days" instead of "d")
    
    RETURNS:
    ------------------------------
    str
        human-readable time
    """

    output = str()
    if seconds < 0:
        seconds = abs(seconds)
        output = "- "

    if seconds < 60:
        small_units = [
            ("second", 1),
            ("millisecond", 1e-3),
            ("microsecond", 1e-6),
            ("nanosecond", 1e-9),
        ] if written_unit else [
            ("s", 1),
            ("ms", 1e-3),
            ("µs", 1e-6),
            ("ns", 1e-9),
        ]

        for name, value in small_units:
            if seconds >= value: # Always show smallest unit
                this_share = seconds / value
                this_share = int(np.ceil(this_share)) if this_share >= 10 else round(this_share, 1)

                if written_unit:
                    output = f"{this_share} {name}{'' if this_share == 1 else 's'}"
                else:
                    output = f"{this_share} {name}"
                break

    else:
        big_units = [
            ("day", 86400),
            ("hour", 3600),
            ("minute", 60),
            ("second", 1),
        ] if written_unit else [
            ("d", 86400),
            ("h", 3600),
            ("m", 60),
            ("s", 1),
        ]

        count_parts = 0
        for name, value in big_units:
            if seconds >= value:
                this_share = int(seconds // value)
                seconds %= value

                if written_unit:
                    output += f"{this_share} {name}{'' if this_share == 1 else 's'} "
                else:
                    output += f"{this_share}{name} "

                count_parts += 1
            
            if count_parts == max_parts:
                break
        
        output = output[:-1]

    return output


def get_cursor_position():
    """
    Retrieves the current cursor position in the terminal.
    """
    # Write the ANSI escape code to query cursor position
    # \033[6n tells the terminal to report the current cursor position
    sys.stdout.write("\033[6n")
    sys.stdout.flush()

    # Read the response from the terminal
    fd = sys.stdin.fileno()  # Get the file descriptor for standard input
    old_settings = termios.tcgetattr(fd)  # Save the current terminal settings
    try:
        tty.setcbreak(fd)  # Set the terminal to raw mode (read input char-by-char)
        response = ""  # Initialize an empty string to store the response
        while True:
            char = sys.stdin.read(1)  # Read one character from stdin
            response += char  # Append the character to the response string
            if char == "R":  # End of the terminal's response (e.g., "\033[5;10R")
                break
    finally:
        # Restore the terminal to its original settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # Parse the response to extract row and column numbers
    try:
        # The response format is "\033[row;columnR"
        _, position = response.split("\033[", 1)  # Remove the initial "\033["
        row, col = map(int, position[:-1].split(";"))  # Split "row;column" and convert to integers
        return row, col  # Return the parsed row and column numbers
    except ValueError:
        # Raise an error if the response cannot be parsed
        raise RuntimeError("Failed to retrieve cursor position.")


def set_linux_terminal_size(width, height):
    try:
        subprocess.run(['resize', '-s', str(height), str(width)])
    except:
        pass

    try:
        subprocess.run(['stty', 'rows', str(height), 'cols', str(width)])
    except:
        pass


def set_macos_terminal_size(width, height):
    applescript = f'''
    tell application "iTerm"
        tell current session of current window
            set columns to {width}
            set rows to {height}
        end tell
    end tell
    '''
    subprocess.run(["osascript", "-e", applescript])


def get_os_type():
    os_type = platform.system()
    if os_type == "Darwin":
        return "macOS"
    elif os_type == "Linux":
        return "Linux"
    else:
        return "Other"


class DynamicProgressBar:
    def __init__(self, total, batch_size: int = 1, seconds_between_updates: float = 1):
        """
        Initializes the dynamic progress bar.

        ARGUMENTS:
        ------------------------------
        total: int
            total number of iterations
        batch_size: int
            batch size (index updates by this number)
        seconds_between_updates: float
            minimum time between updates

        RETURNS:
        ------------------------------
        None
        """
        self.start_time = time.time()
        self.min_time_between_updates = seconds_between_updates
        self.last_bar_update = self.start_time - self.min_time_between_updates

        self.padding_right = 3
        self.total = total
        self.batch_size = batch_size
        self.auto_index = 0

        terminal_width = shutil.get_terminal_size().columns

        initial_message = "Initializing progress bar..."
        if len(initial_message) > terminal_width:
            initial_message = "Init. progress bar..."
            if len(initial_message) > terminal_width:
                initial_message = "Init. prog. bar..."
                if len(initial_message) > terminal_width:
                    initial_message = "Init. pb."
                    if len(initial_message) > terminal_width:
                        initial_message = "Init."
                        if len(initial_message) > terminal_width:
                            initial_message = ""

        sys.stdout.write(initial_message)
        sys.stdout.flush()
        
        self.previous_output_length = len(initial_message)

        # determine if running inside jupyter notebook (ANSI escape codes do not work there)
        self.jupyter_notebook = False
        try:
            if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
                self.jupyter_notebook = True
        except NameError:
            pass
    

    def _design_progress_bar(
            self,
            index: int,
            terminal_width: int = 0,
            additional_info: str = ""
        ):
        """
        Prints a progress bar to the console.

        Idea taken from:
        https://stackoverflow.com/questions/6169217/replace-console-output-in-python

        ARGUMENTS:
        ------------------------------
        index: int
            current index
        terminal_width: int
            width of the terminal
        additional_info: str
            additional information to print at the end of the progress bar
        

        RETURNS:
        ------------------------------
        current_length: int
            length of the current message generated by this function
        """

        # calculate time passed, time per index and estimated total time
        if index == 0:
            time_message = ''
        else:
            time_passed = time.time() - self.start_time
            time_per_index = time_passed/index
            time_total = time_per_index*self.total
            # time_remaining = time_per_index*(total-index)
            time_per_index *= self.batch_size

            time_passed_str = format_time(time_passed)
            time_per_index_str = format_time(time_per_index)
            # time_remaining_str = format_time(time_remaining)
            time_total_str = format_time(time_total)

            time_message = f' | {time_passed_str} / {time_total_str} ({time_per_index_str}/it)'
        
        # calculate percentage done
        percent_done = index/self.total*100
        rounded_percent_done = round(percent_done, 1)

        if index >= self.total:
            basic_message = f'   ✅: {rounded_percent_done}%'
        else:
            basic_message = f'   ⏳: {rounded_percent_done}%'
        done_message = f' {index} / {self.total}'
        additional_message = f' | {additional_info}'

        # evaluate bar length
        remaining_length = terminal_width - len(basic_message) - 3 - len(done_message) - len(time_message) - len(additional_message)
        bar_length = remaining_length
        if remaining_length < 0:
            bar_length = 0

        # build done and to-go part of the bar
        done = round(percent_done/100*bar_length)
        togo = bar_length-done

        done_str = '█'*int(done)
        togo_str = '░'*int(togo)

        # based upon available length, decide what to print
        if remaining_length >= 0:
            message = basic_message + f' [{done_str}{togo_str}]' + done_message + time_message + additional_message
        elif remaining_length + len(additional_message) >= 0:
            message = basic_message + ' []' + done_message + time_message
        elif remaining_length + len(additional_message) + len(done_message) >= 0:
            message = basic_message + ' []' + time_message
        else:
            message = basic_message
        
        # remove blank spaces at the end of the message
        for i in range(len(message)-1, -1, -1):
            if message[i] != ' ':
                break

        return message[:i+1]
    

    def _generate_clearing_sequence(self, current_terminal_width):
        # if terminal size was reduced since last print, the previous output is now displayed on multiple 
        # lines, calculate on how many:
        lines_to_clear = self.previous_output_length / current_terminal_width
        lines_to_clear = int(lines_to_clear) + 1
        
        # build clearing sequence
        clearing_sequence = ""

        # clear previous output
        for _ in range(lines_to_clear):
            clearing_sequence += "\033[2K" # Clear line
            clearing_sequence += "\033[F" # Move cursor up

        clearing_sequence += "\n" # Move cursor to next line
        
        return clearing_sequence


    def update(self, current_index=None, additional_info=""):
        # update auto index
        self.auto_index += self.batch_size
        
        # assign auto index to current index if not provided
        if current_index is None:
            current_index = self.auto_index
        
        # limit current index to total
        if current_index > self.total:
            current_index = self.total

        # skip update if not enough time has passed since last update
        if current_index != self.total:
            if time.time() - self.last_bar_update < self.min_time_between_updates:
                return

        while True:
            current_terminal_width = shutil.get_terminal_size().columns # additional space on the right

            # design progress bar
            progress_bar = self._design_progress_bar(
                index = current_index,
                terminal_width = current_terminal_width - self.padding_right,
                additional_info = additional_info,
            )

            # generate clearing sequence using ANSI escape codes
            clearing_sequence = self._generate_clearing_sequence(current_terminal_width)
            
            # if running inside jupyter notebook, ANSI excape codes do not work, but output is not affected by terminal size
            if self.jupyter_notebook:
                clearing_sequence = "\r"

            # paste the string to execute clearing sequence and build progress bar to one output
            output = clearing_sequence + progress_bar

            # execute clearing sequence and build progress bar only if terminal size was not altered during the process
            if shutil.get_terminal_size().columns == current_terminal_width:
                sys.stdout.write(output)
                sys.stdout.flush()

                self.previous_output_length = len(progress_bar)
                break
        
        # perform line break if progress bar is finished
        if current_index == self.total:
            print("")

        # update last bar update time
        self.last_bar_update = time.time()


def handle_resize(signum, frame):
    time.sleep(1)

# uncomment to sleep during terminal resize (important for updating progress bar without visual errors during high speed resizing)
# signal.signal(signal.SIGWINCH, handle_resize) # increases computation time by factor 1/216


"""
===============
Override Files
===============
"""


def retrieve_user_response(message: str, allowed_responses: list):
    """
    Prints message to console and retrieves user response.

    ARGUMENTS:
    ------------------------------
    message: str
        message to print
    allowed_responses: list
        list of allowed answers
    
    RETURNS:
    ------------------------------
    str
        user response
    """

    while True:
        answer = input("\n" + message)
        
        if answer in allowed_responses:
            return answer
        else:
            print(f"\nPlease enter one of the following: {allowed_responses}")


def delete_files(file_paths: list):
    """
    Deletes files.

    ARGUMENTS:
    ------------------------------
    file_paths: list
        list of file paths to delete
    
    RETURNS:
    ------------------------------
    None
    """

    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)


def delete_directory_files(directory_path: str, keep_files: list):
    """
    Deletes all files in a directory.

    ARGUMENTS:
    ------------------------------
    directory_path: str
        path to directory
    keep_files: list
        list of file names to keep
    
    RETURNS:
    ------------------------------
    None
    """

    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        if file in keep_files or file_path in keep_files:
            continue
        if os.path.isfile(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    # total = 100
    # bar = DynamicProgressBar(total=total, batch_size=1, seconds_between_updates=0)

    # for i in range(total):
    #     time.sleep(0.05)
    #     bar.update()
    
    # print(bar.auto_index)

    total = 100
    bar = DynamicProgressBar(total=total, batch_size=10, seconds_between_updates=0)

    for i in range(10):
        time.sleep(0.15)
        bar.update()
    
    print(bar.auto_index)