from re import findall

def extract_numbers(input_string) -> str:
    # Find all sequences of digits in the string
    numbers = findall(r'\d+', input_string)
    return numbers[0]
