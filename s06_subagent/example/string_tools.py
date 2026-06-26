"""String utility functions."""

import re
import unicodedata


def slugify(text: str) -> str:
    """Convert a string into a URL-friendly slug.

    Steps performed:
        1. Lowercase the text.
        2. Strip leading/trailing whitespace.
        3. Normalise Unicode characters and convert accented characters
           to their ASCII equivalents (e.g. 'é' -> 'e', 'ñ' -> 'n').
        4. Replace any remaining non-alphanumeric characters (except
           hyphens) with a single hyphen.
        5. Collapse multiple consecutive hyphens into one hyphen.
        6. Strip leading/trailing hyphens.

    Args:
        text: The input string to slugify.

    Returns:
        A clean, hyphen-separated slug string.
    """
    # Step 1: Lowercase
    result = text.lower()

    # Step 2: Strip leading/trailing whitespace
    result = result.strip()

    # Step 3: Normalise Unicode and convert accented chars to ASCII
    #   NFKD decomposes characters like 'é' into 'e' + combining accent,
    #   then we keep only ASCII characters (ord(c) < 128).
    result = unicodedata.normalize('NFKD', result)
    result = result.encode('ascii', 'ignore').decode('ascii')

    # Step 4: Replace non-alphanumeric characters (except hyphens) with hyphens
    result = re.sub(r'[^a-z0-9-]', '-', result)

    # Step 5: Collapse multiple consecutive hyphens into one
    result = re.sub(r'-+', '-', result)

    # Step 6: Strip leading/trailing hyphens
    result = result.strip('-')

    return result


if __name__ == '__main__':
    import sys

    # Simple test cases
    tests = [
        ("Hello World", "hello-world"),
        ("  Leading and trailing  ", "leading-and-trailing"),
        ("Special !!! chars *** here", "special-chars-here"),
        ("Already-a-slug", "already-a-slug"),
        ("Caf\u00e9 au\u00f1ita", "cafe-aunita"),
        ("  ---Multiple---Hyphens---  ", "multiple-hyphens"),
        ("", ""),
        ("-", ""),
    ]

    all_pass = True
    for input_str, expected in tests:
        output = slugify(input_str)
        status = "PASS" if output == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        # Use safe ASCII repr to avoid encoding issues on Windows
        safe_input = input_str.encode('unicode_escape').decode('ascii')
        print(f"[{status}] slugify({safe_input}) = {output!r} (expected {expected!r})")

    sys.exit(0 if all_pass else 1)
