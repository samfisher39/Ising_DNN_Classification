def get_nearest_proper_divisor(_divisor, _number):
    """
    sequentially checks the next higher/lower number e.g. 10->9->11->8->12...
    if it is a proper divisor of _number

    :param _divisor: (int) number of which the next nearest proper divisor of _number is wanted
    :param _number: (int) an arbitrary int
    :return: (int) next nearest divisor to _divisor of _number
    """
    if _divisor > _number or _divisor < 0:
        raise ValueError("The following statement is not fulfilled 0 < divisor < number!")

    if _number % _divisor == 0:
        return _divisor
    else:
        for i in range(1, _number):
            _divisor = (_divisor + (-1) ** i * i)
            if _number % _divisor == 0:
                return _divisor
    print("Error in method 'get_nearest_proper_divisor', found no divisor of %i" % _number)
