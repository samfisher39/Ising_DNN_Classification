PROMPT = "#"
info_box_width = 60


def bar():
    """
    print a horizontal bar/ruler of the specified PROMPT string
    :return: string
    """
    times = info_box_width // len(PROMPT) - 1
    return " " + times * PROMPT


def print_hline():
    times = info_box_width // len(PROMPT) - 5
    print_log("", _bottom=False, _top=False, log_file="./logs/latest.txt")
    print_log(" " + times * "~" + " ", _bottom=False, _top=False, log_file="./logs/latest.txt")
    print_log("", _bottom=False, _top=False, log_file="./logs/latest.txt")


def print_log(*_args, _top=True, _bottom=True, log_file=None):
    """
    print information on the calculations to the screen and to a log file.
    :param _args: list of strings to be printed to the screen
    :param _top: whether the first line should be a horizontal bar
    :param _bottom: whether the last line should be a horizontal bar
    :param log_file: file which the log is printed to
    :return: None
    """
    if log_file is not None:
        f = open(log_file, "a+")

    if _top:
        print(bar())
        if log_file is not None:
            f.write(bar() + "\n")

    for i, line in enumerate(_args):
        length = len(line)
        if length < info_box_width:
            space = info_box_width - length - 2 * len(PROMPT)
            if space % 2 == 0:
                line_printed = PROMPT + (space // 2 - 1) * " " + line + space // 2 * " " + PROMPT[::-1]
            else:
                line_printed = PROMPT + space // 2 * " " + line + space // 2 * " " + PROMPT[::-1]
        else:
            overflow = length - info_box_width
            if overflow % 2 == 0:
                line_printed = PROMPT + " " + line[: -overflow - 2 * len(PROMPT) - 6] + "... " + PROMPT[::-1]
            else:
                line_printed = PROMPT + " " + line[: -overflow - 2 * len(PROMPT) - 5] + "... " + PROMPT[::-1]

        print(" " + line_printed)
        if log_file is not None:
            f.write(" " + line_printed + "\n")

    if _bottom:
        print(bar())
        if log_file is not None:
            f.write(bar() + "\n")

    if log_file is not None:
        f.close()
