"""
Subtitle processing utilities for Qwen3-ASR.

Provides:
  - SRT timestamp formatting
  - CJK-aware token joining (no spaces for CJK characters, spaces for Latin)
  - Grouping word-level forced-aligner timestamps into subtitle segments
"""


def format_srt_time(seconds: float) -> str:
    """Convert a float seconds value to SRT timestamp format HH:MM:SS,mmm."""
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def join_tokens(a: str, b: str) -> str:
    """Join two text tokens, omitting the space when either boundary character is CJK.

    Note: The CJK detection covers the CJK Unified Ideographs block (U+4E00–U+9FFF),
    which handles the most common Chinese characters.  Hiragana, Katakana, Hangul, and
    CJK Extension blocks are not included (spaces are preserved for those scripts).
    """
    if not a:
        return b
    if not b:
        return a
    # Don't insert a space when either side touches a CJK Unified Ideograph character.
    for ch in (a[-1], b[0]):
        if "\u4e00" <= ch <= "\u9fff":
            return f"{a}{b}"
    return f"{a} {b}"


def group_time_stamps(time_stamps, max_gap_sec: float, max_chars: int, split_mode: str):
    """
    Group word-level forced-aligner timestamps into subtitle line dicts.

    Args:
        time_stamps:   Iterable of objects with .text, .start_time, .end_time attributes.
        max_gap_sec:   Silence gap (seconds) that triggers a new subtitle line when
                       ``pause`` is included in *split_mode*.
        max_chars:     Maximum character count for a subtitle line when ``length`` is
                       included in *split_mode*.  0 disables the length check.
        split_mode:    Strategy string; checked via substring containment, so it may
                       include any combination of the keywords ``punctuation``, ``pause``,
                       and ``length`` (e.g. ``"punctuation_pause_length"`` enables all
                       three strategies).  The default used in the API is
                       ``"split_by_punctuation_or_pause_or_length"``, which contains all
                       three keywords and therefore enables every strategy.

    Returns:
        List of dicts with keys ``start``, ``end``, ``text``.
    """
    if not time_stamps:
        return []

    groups = []
    cur = None
    punct = ("。", "！", "？", ".", "!", "?")

    for item in time_stamps:
        text = (item.text or "").strip()
        if not text:
            continue
        if cur is None:
            cur = {"start": item.start_time, "end": item.end_time, "text": text}
            continue

        gap = float(item.start_time) - float(cur["end"])
        too_far = gap > max_gap_sec
        too_long = max_chars > 0 and (len(cur["text"]) + len(text)) > max_chars
        end_sentence = any(cur["text"].endswith(p) for p in punct)

        should_split = False
        if "punctuation" in split_mode and end_sentence:
            should_split = True
        if "length" in split_mode and too_long:
            should_split = True
        if "pause" in split_mode and too_far:
            should_split = True

        if should_split:
            groups.append(cur)
            cur = {"start": item.start_time, "end": item.end_time, "text": text}
        else:
            cur["text"] = join_tokens(cur["text"], text).strip()
            cur["end"] = item.end_time

    if cur is not None:
        groups.append(cur)
    return groups
