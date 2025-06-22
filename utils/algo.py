from typing import Callable, List, Tuple
from utils.maths import compute_approximation_error
from tqdm.auto import tqdm

def optimal_approx(
    n: int,
    f: Callable,
    a: float,
    b: float,
    stepsize: float
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Compute the optimal approximation of a convex function.

    Parameters:
    - n: The number of points to use.
    - f: The function to approximate.
    - a: The start of the interval.
    - b: The end of the interval.
    - stepsize: The stepsize to use for the approximation.

    Returns:
    - Tuple of points and the optimal approximation error.
    """

    # Initialize intervals equally distributed
    intervals = [(a + i * (b - a) / n, a + (i + 1) * (b - a) / n) for i in range(n)]

    def compute_segment_error(segment: Tuple[float, float]) -> float:
        """Compute error for a single segment using the custom utility."""
        result = compute_approximation_error(f, segment)
        return result["error"]

    # Compute initial errors
    errors = [compute_segment_error(interval) for interval in intervals]
    rounds = 0

    while True:
        max_error = max(errors)
        min_error = min(errors)
        prev_error_diff = max_error - min_error

        for i in range(1, n):
            left_error = errors[i - 1]
            right_error = errors[i]
            x_common = intervals[i][0]  # common endpoint

            if left_error > right_error:
                while left_error > right_error:
                    rounds += 1
                    x_common -= stepsize
                    intervals[i - 1] = (intervals[i - 1][0], x_common)
                    intervals[i] = (x_common, intervals[i][1])
                    left_error = compute_segment_error(intervals[i - 1])
                    right_error = compute_segment_error(intervals[i])
            elif left_error < right_error:
                while left_error < right_error:
                    rounds += 1
                    x_common += stepsize
                    intervals[i - 1] = (intervals[i - 1][0], x_common)
                    intervals[i] = (x_common, intervals[i][1])
                    left_error = compute_segment_error(intervals[i - 1])
                    right_error = compute_segment_error(intervals[i])

        # Recompute errors
        errors = [compute_segment_error(interval) for interval in intervals]
        max_error = max(errors)
        min_error = min(errors)
        current_error_diff = max_error - min_error

        # Check convergence
        if prev_error_diff <= current_error_diff:
            break

    return intervals, errors, rounds

def run_optimal_approximation(n, func, func_name, a, b, step, errs, get_derivative, mean_err, max_fx, min_fx):
    for i in tqdm(n, desc=f"Optimal Approximation: {func_name}"):
        print(f"n = {i}")
        print(f"f(x) = {func_name}")
        a, b = 0, 1
        alpha = (b-a)**2 / (16 * i**2)
        res = optimal_approx(i, func, a, b, step)
        errs.append({
            "function": func_name,
            "n": i,
            "mean_err": mean_err(res[1]),
            "upper_bound": max_fx(get_derivative(get_derivative(func)), (a, b)) * alpha,
            "lower_bound": min_fx(get_derivative(get_derivative(func)), (a, b)) * alpha,
            "max_gap": max(res[1]) - min(res[1]),
            "rounds": res[2]
        })
        print('-'*50)
