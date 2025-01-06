def levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein distance (edit distance) between two sequences (strings or lists).
    """

    if len(seq1) == 0:
        return len(seq2)
    if len(seq2) == 0:
        return len(seq1)

    dp = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]

    for i in range(len(seq1) + 1):
        dp[i][0] = i
    for j in range(len(seq2) + 1):
        dp[0][j] = j

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1  # 0 if matching, 1 if substitution
            dp[i][j] = min(
                dp[i - 1][j] + 1, # deletion
                dp[i][j - 1] + 1, # insertion
                dp[i - 1][j - 1] + cost  # substitution (cost=0 if same char/word)
            )

    return dp[-1][-1]


def calc_cer(target_text, predicted_text) -> float:

    if len(target_text) == 0 and len(predicted_text) == 0:
        return 0.0

    distance = levenshtein_distance(target_text, predicted_text)
    denominator = max(len(target_text), 1)

    return distance / denominator


def calc_wer(target_text, predicted_text) -> float:
    target_words = target_text.split()
    predicted_words = predicted_text.split()

    if len(target_words) == 0 and len(predicted_words) == 0:
        return 0.0

    distance = levenshtein_distance(target_words, predicted_words)
    denominator = max(len(target_words), 1)

    return distance / denominator
