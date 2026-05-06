import math


LETTERS = ("B", "K", "O")
END = "-"

TRANSITIONS = {
    "B": {"B": 0.1, "K": 0.325, "O": 0.25, END: 0.325},
    "K": {"B": 0.4, "K": 0.0, "O": 0.4, END: 0.2},
    "O": {"B": 0.2, "K": 0.2, "O": 0.2, END: 0.4},
    END: {"B": 1.0, "K": 0.0, "O": 0.0, END: 0.0},
}


def safe_log(probability):
    if probability == 0:
        return -math.inf
    return math.log(probability)


def most_probable_word(length):
    if length < 1:
        raise ValueError("Word length must be at least 1")

    # Every word starts with B with probability 1.
    dp = {"B": (0.0, "B")}

    for _ in range(2, length + 1):
        next_dp = {}
        for next_letter in LETTERS:
            best_log_probability = -math.inf
            best_word = None

            for previous_letter, (log_probability, word) in dp.items():
                transition_log_probability = safe_log(
                    TRANSITIONS[previous_letter][next_letter]
                )
                candidate_log_probability = (
                    log_probability + transition_log_probability
                )

                if candidate_log_probability > best_log_probability:
                    best_log_probability = candidate_log_probability
                    best_word = word + next_letter

            next_dp[next_letter] = (best_log_probability, best_word)

        dp = next_dp

    best_log_probability = -math.inf
    best_word = None

    for last_letter, (log_probability, word) in dp.items():
        candidate_log_probability = log_probability + safe_log(
            TRANSITIONS[last_letter][END]
        )

        if candidate_log_probability > best_log_probability:
            best_log_probability = candidate_log_probability
            best_word = word

    return best_word, math.exp(best_log_probability)


def word_probability(word):
    if not word or word[0] != "B":
        return 0.0

    probability = 1.0
    for current_letter, next_letter in zip(word, word[1:]):
        probability *= TRANSITIONS[current_letter][next_letter]
    probability *= TRANSITIONS[word[-1]][END]
    return probability


def main():
    length = 5
    word, probability = most_probable_word(length)

    print("Most probable word of size %d: %s" % (length, word))
    print("Probability: %.8f" % probability)


if __name__ == "__main__":
    main()
