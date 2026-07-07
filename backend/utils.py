import numpy as np


def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))


def eye_aspect_ratio(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR).

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        eye_landmarks: List of 6 (x, y) tuples for eye landmarks.

    Returns:
        float: EAR value
    """
    p1, p2, p3, p4, p5, p6 = eye_landmarks

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return round(float(ear), 4)


def mouth_aspect_ratio(mouth_landmarks):
    """
    Calculate Mouth Aspect Ratio (MAR).

    MAR = vertical mouth opening / horizontal mouth width

    Args:
        mouth_landmarks: List of 4 (x, y) tuples [top, bottom, left, right]

    Returns:
        float: MAR value
    """
    top, bottom, left, right = mouth_landmarks

    vertical = euclidean_distance(top, bottom)
    horizontal = euclidean_distance(left, right)

    if horizontal == 0:
        return 0.0

    mar = vertical / horizontal
    return round(float(mar), 4)


def smooth_value(history, new_value, max_len=5):
    """
    Smooth a metric value using a rolling average.

    Args:
        history: list of previous values
        new_value: latest value
        max_len: window size

    Returns:
        tuple: (smoothed_value, updated_history)
    """
    history.append(new_value)
    if len(history) > max_len:
        history.pop(0)
    return round(sum(history) / len(history), 4), history
