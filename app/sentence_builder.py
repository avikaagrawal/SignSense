from collections import deque, Counter

from config import (
    MAX_HISTORY,
    MIN_HISTORY,
    STABILITY_THRESHOLD,
    CONFIDENCE_THRESHOLD
)


class SentenceBuilder:

    def __init__(self):

        self.history = deque(maxlen=MAX_HISTORY)
        self.sentence = ""

        self.last_letter = None

        # True when we are allowed to accept a sign
        self.ready_for_next = True

    def reset(self):

        self.history.clear()

        # Losing the hand completely re-arms the system
        self.last_letter = None
        self.ready_for_next = True

    def update(self, letter, confidence):

        prediction_text = "Detecting..."

        # Ignore low-confidence predictions
        if confidence < CONFIDENCE_THRESHOLD:
            return prediction_text

        self.history.append(letter)

        # Not enough frames yet
        if len(self.history) < MIN_HISTORY:
            return prediction_text

        # Find the most common recent prediction
        stable_letter, count = Counter(
            self.history
        ).most_common(1)[0]

        stability = count / len(self.history)

        # Prediction isn't stable
        if stability < STABILITY_THRESHOLD:
            return prediction_text

        prediction_text = stable_letter

        # ---------------------------------
        # SAME SIGN IS STILL BEING HELD
        # ---------------------------------

        if stable_letter == self.last_letter:

            # Keep waiting.
            # Don't add the same letter again.
            return prediction_text

        # ---------------------------------
        # DIFFERENT SIGN / NEW SIGN
        # ---------------------------------

        self._add_character(stable_letter)

        self.last_letter = stable_letter

        return prediction_text

    def _add_character(self, letter):

        # SPACE
        if letter == "space":

            self.sentence += " "

        # DELETE
        elif letter == "del":

            if self.sentence:
                self.sentence = self.sentence[:-1]

        # NORMAL LETTER
        else:

            self.sentence += letter