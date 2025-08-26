import pandas as pd
import numpy as np
import math
import itertools


def safe_fmt(x):
    if x is None or pd.isna(x):
        return ""
    return f"{x:.4f}"


def bootstrap_ci_of_diff(x, y=None, n_resamples=10000, alpha=0.05):
    """
    Bootstrap CI for mean of x (if y is None) or mean of x - y.
    If total unique bootstrap samples <= 10000, enumerate all.
    Otherwise, do random bootstrap.
    """
    vec = x if y is None else x - y
    n = len(vec)

    # Total number of unique bootstrap samples (combinations with replacement)
    n_boot_combinations = math.comb(n + n - 1, n)

    if n_boot_combinations <= n_resamples:
        indices = itertools.combinations_with_replacement(range(n), n)
        boot_stats = [np.mean(vec[list(idx)]) for idx in indices]
    else:
        boot_stats = [
            np.mean(np.random.choice(vec, size=n, replace=True))
            for _ in range(n_resamples)
        ]

    lower = np.percentile(boot_stats, 100 * (alpha / 2))
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return lower, upper


def permutation_p_value(x, y=None, n_resamples=10000):
    """
    Paired permutation test for mean difference using exact sign flips
    if 2^n ≤ 10000; otherwise falls back to random permutations.
    """
    vec = x if y is None else x - y
    n = len(vec)
    n_permutations = 2**n  # total unique sign flips

    observed_mean = np.mean(vec)

    if n_permutations <= n_resamples:
        # Generate all sign flips
        signs = np.array(list(itertools.product([-1, 1], repeat=n)))
        permuted_means = np.mean(signs * vec, axis=1)
    else:
        # Randomly sample sign flips
        permuted_means = [
            np.mean(vec * np.random.choice([-1, 1], size=n)) for _ in range(n_resamples)
        ]

    p = np.mean(np.abs(permuted_means) >= np.abs(observed_mean))

    return p


def combined_significance_test(
    timesfm_scores, random_scores, alpha=0.05, round_digits=2
):
    """
    Test if TimesFM is significantly better than both random and zero.
    Returns detailed results for both individual tests and combined interpretation.
    """
    timesfm_scores = np.round(timesfm_scores, round_digits)
    random_scores = np.round(random_scores, round_digits)
    # Test 1: TimesFM vs Random
    p_vs_random = permutation_p_value(timesfm_scores, random_scores)

    # Test 2: TimesFM vs Zero
    p_vs_zero = permutation_p_value(timesfm_scores)

    # Combined test: Both must be significant
    is_significant = (p_vs_random < alpha) and (p_vs_zero < alpha)

    # For CI, we want the difference from random (more interpretable)
    ci_low, ci_high = bootstrap_ci_of_diff(timesfm_scores, random_scores)

    # Check for degenerate cases
    is_all_zero = np.all(timesfm_scores == 0)
    is_all_one = np.all(timesfm_scores == 1)

    return {
        "p_vs_random": p_vs_random,
        "p_vs_zero": p_vs_zero,
        "significant_vs_random": p_vs_random < alpha,
        "significant_vs_zero": p_vs_zero < alpha,
        "overall_significant": is_significant,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mean_diff": np.mean(timesfm_scores - random_scores),
        "is_all_zero": is_all_zero,
        "is_all_one": is_all_one,
    }


def format_result_with_symbols(row):
    """Format results with clear symbols showing what failed"""
    p_rand = row["p_vs_random"]
    p_zero = row["p_vs_zero"]
    sig_rand = row["significant_vs_random"]
    sig_zero = row["significant_vs_zero"]
    overall_sig = row["overall_significant"]
    is_all_zero = row["is_all_zero"]
    is_all_one = row["is_all_one"]
    ci_low = row["ci_low"]
    ci_high = row["ci_high"]

    # Format p-values
    def fmt_p(p):
        if pd.isna(p):
            return "-"
        if p < 0.001:
            return "$<$0.001"
        return f"{p:.3f}"

    # Get significance stars
    def get_stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    # Format CI
    def fmt_ci(low, high, threshold=1e-3):
        if pd.isna(low) or pd.isna(high):
            return "-"
        if abs(low) < threshold:
            low = 0.0
        if abs(high) < threshold:
            high = 0.0
        low_str = f"{low:.3f}".replace("-0.000", "0.000")
        high_str = f"{high:.3f}".replace("-0.000", "0.000")
        return f"[{low_str}, {high_str}]"

    # Build result string with symbols
    if overall_sig:
        # Both tests significant - show the WORSE (more conservative) p-value
        p_display = max(p_rand, p_zero)
        stars = get_stars(p_display)
        p_str = f"{fmt_p(p_display)}$^{{{stars}}}$"
    else:
        # NOT truly significant - show failure symbols
        symbols = []
        if not sig_rand:
            symbols.append("$^{\\nsim R}$")  # Not significantly different from Random
        if not sig_zero:
            symbols.append("$^{\\nsim 0}$")  # Not significantly different from Zero

        # For failed cases, show the WORSE p-value (higher = less significant)
        # This prevents misleading significance claims
        p_display = max(p_rand, p_zero)

        # NO stars for failed cases since overall result is not significant
        p_str = f"{fmt_p(p_display)}{''.join(symbols)}"

    # Add degenerate case symbols
    ci_str = fmt_ci(ci_low, ci_high)
    if is_all_zero:
        ci_str += "$^{\\dagger}$"  # All predictions are 0
    elif is_all_one:
        ci_str += "$^{\\ddagger}$"  # All predictions are 1

    return p_str, ci_str
