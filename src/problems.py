PROBLEMS = [
    {
        "id": 1,
        "problem": "Write a Python function `has_close_elements(numbers, threshold)` that checks if any two numbers in a list are closer to each other than a given threshold.",
        "test": (
            "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n"
            "assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n"
            "assert has_close_elements([], 1.0) == False"
        ),
    },
    {
        "id": 2,
        "problem": "Write a Python function `remove_duplicates(numbers)` that removes all elements from a list that appear more than once, keeping the original order.",
        "test": (
            "assert remove_duplicates([1, 2, 3, 2, 4]) == [1, 3, 4]\n"
            "assert remove_duplicates([1, 1, 1]) == []\n"
            "assert remove_duplicates([1, 2, 3]) == [1, 2, 3]"
        ),
    },
    {
        "id": 3,
        "problem": "Write a Python function `count_distinct_pairs(lst, diff)` that counts the number of distinct pairs (a, b) where a < b and b - a == diff.",
        "test": (
            "assert count_distinct_pairs([1, 5, 3, 4, 2], 3) == 2\n"
            "assert count_distinct_pairs([1, 2, 3, 4, 5], 1) == 4\n"
            "assert count_distinct_pairs([1, 1, 1], 0) == 0"
        ),
    },
    {
        "id": 4,
        "problem": "Write a Python function `flatten(nested)` that takes a nested list of arbitrary depth and returns a flat list.",
        "test": (
            "assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]\n"
            "assert flatten([]) == []\n"
            "assert flatten([[1, 2], [3, [4, 5]]]) == [1, 2, 3, 4, 5]"
        ),
    },
    {
        "id": 5,
        "problem": "Write a Python function `longest_palindrome(s)` that finds the longest palindromic substring in a string.",
        "test": (
            "assert longest_palindrome('babad') in ('bab', 'aba')\n"
            "assert longest_palindrome('cbbd') == 'bb'\n"
            "assert longest_palindrome('a') == 'a'"
        ),
    },
    {
        "id": 6,
        "problem": "Write a Python function `group_by_first_letter(words)` that takes a list of strings and returns a dict mapping each first letter to the list of words starting with that letter, preserving order.",
        "test": (
            "assert group_by_first_letter(['apple', 'avocado', 'banana', 'blueberry']) == {'a': ['apple', 'avocado'], 'b': ['banana', 'blueberry']}\n"
            "assert group_by_first_letter([]) == {}\n"
            "assert group_by_first_letter(['cat']) == {'c': ['cat']}"
        ),
    },
    {
        "id": 7,
        "problem": "Write a Python function `running_median(nums)` that returns a list of medians after each element is added to a running sequence.",
        "test": (
            "assert running_median([2, 1, 5, 7, 2, 0, 5]) == [2, 1.5, 2, 3.5, 2, 2.0, 2]\n"
            "assert running_median([1]) == [1]\n"
            "assert running_median([3, 1]) == [3, 2.0]"
        ),
    },
    {
        "id": 8,
        "problem": "Write a Python function `first_repeated_char(s)` that returns the first character in a string that appears more than once, or None if there is no repeated character.",
        "test": (
            "assert first_repeated_char('abcabc') == 'a'\n"
            "assert first_repeated_char('abcdef') is None\n"
            "assert first_repeated_char('aabbcc') == 'a'"
        ),
    },
]
