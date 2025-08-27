def compress(data, window_size=20, lookahead_buffer_size=15):
    """
    Compresses data using the LZ77 algorithm.

    Args:
        data: The input string to compress.
        window_size: The size of the sliding window.
        lookahead_buffer_size: The size of the lookahead buffer.

    Returns:
        A list of tuples representing the compressed data.
        Each tuple is (offset, length, next_char).
    """
    compressed_data = []
    i = 0
    while i < len(data):
        match_offset = 0
        match_length = 0
        # Search for the longest match in the sliding window
        # The search window is data[max(0, i - window_size):i]
        # The lookahead buffer is data[i:min(len(data), i + lookahead_buffer_size)]

        search_buffer_start = max(0, i - window_size)
        search_buffer = data[search_buffer_start:i]
        
        lookahead_buffer_end = min(len(data), i + lookahead_buffer_size)

        for length in range(min(lookahead_buffer_size, len(data) - i), 0, -1):
            substring_to_match = data[i:i+length]
            # Find the last occurrence to favor longer distances for shorter matches
            # if multiple matches of the same length exist.
            offset_in_search_buffer = search_buffer.rfind(substring_to_match)

            if offset_in_search_buffer != -1:
                match_offset = len(search_buffer) - offset_in_search_buffer
                match_length = length
                break
        
        if match_length > 0:
            # Check if there's a next character after the match
            if i + match_length < len(data):
                next_char = data[i + match_length]
            else:
                # No next character, it's the end of the data
                next_char = '' 
            compressed_data.append((match_offset, match_length, next_char))
            i += match_length + 1
        else:
            # No match found, output (0, 0, current_char)
            compressed_data.append((0, 0, data[i]))
            i += 1
            
    return compressed_data

def decompress(compressed_data):
    """
    Decompresses data compressed by the LZ77 algorithm.

    Args:
        compressed_data: A list of tuples (offset, length, next_char).

    Returns:
        The decompressed string.
    """
    decompressed_data = []
    for offset, length, char in compressed_data:
        if length == 0 and offset == 0: # Uncompressed character
            decompressed_data.append(char)
        else:
            # Calculate the start of the segment to copy
            # The segment is in the already decompressed part of the data
            start_index = len(decompressed_data) - offset
            for _ in range(length):
                decompressed_data.append(decompressed_data[start_index])
                start_index += 1
            if char: # If there's a next character, append it
                 decompressed_data.append(char)
    return "".join(decompressed_data)

if __name__ == '__main__':
    # Example with "Hello World"
    print("\n--- Compressing: 'Hello World' ---")
    hw_data = "Hello World"
    print(f"Original data: \"{hw_data}\"")

    compressed_hw = compress(hw_data) # Using default window/lookahead sizes
    print("\nCompressed output (offset, length, next_char):")
    if not compressed_hw:
        print("  (No output)")
    for token in compressed_hw:
        print(f"  {token}")

    # Decompress silently to verify, but don't print explanation or result unless failed
    decompressed_hw = decompress(compressed_hw)
    assert hw_data == decompressed_hw, "INTERNAL ERROR: Decompression for 'Hello World' failed!"
    print("--- End of 'Hello World' example ---")
