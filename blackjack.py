def hand_total(hand):
    """Helper function to calculate the total points of a blackjack hand.
    """
    total = 0
    # Count the number of aces and deal with how to apply them at the end.
    aces = 0
    for card in hand:
        if card in ['J', 'Q', 'K']:
            total += 10
        elif card == 'A':
            aces += 1
        else:
            # Convert number cards (e.g. '7') to ints
            total += int(card)
    # At this point, total is the sum of this hand's cards *not counting aces*.

    # Add aces, counting them as 1 for now. This is the smallest total we can make from this hand
    total += aces
    # "Upgrade" aces from 1 to 11 as long as it helps us get closer to 21
    # without busting
    while total + 10 <= 21 and aces > 0:
        # Upgrade an ace from 1 to 11
        total += 10
        aces -= 1
    return total

def blackjack_hand_greater_than(hand_1, hand_2):
    total_1 = hand_total(hand_1)
    total_2 = hand_total(hand_2)
    return total_1 <= 21 and (total_1 > total_2 or total_2 > 21)

def partition(arr):

    l = len(arr) - 1 # Last index in array
    pivot = arr[l] # Last element
    i = -1 # Pointer

    for j in range(l): # Loop but don't check the last element
        if arr[j] <= pivot: # Check if value is right to pivot
            i += 1 # Increment pointer
            (arr[i], arr[j]) = (arr[j], arr[i]) # Swap new pointer index with value

    (arr[i+1], arr[l]) = (arr[l], arr[i+1]) # Swap last pointer with Pivot
    return i+1 # Correct position of Pivot

def sort(arr): # Quick sort
    high = len(arr) - 1
    x = partition(arr)
    sort(arr, low, x - 1)
    sort(arr, x + 1, high)














