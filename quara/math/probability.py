import numpy as np

start_red = "\033[31m"
end_color = "\033[0m"


def validate_prob_dist(
    prob_dist: np.ndarray,
    eps: float = None,
    validate_sum: bool = True,
    raise_error: bool = True,
    message: str = "",
) -> None:
    """validate the probability distribution.

    Parameters
    ----------
    prob_dist : np.ndarray
        the probability distribution.
    eps : float, optional
        the absolute tolerance parameter, by default 1e-8.
        checks ``absolute(the sum of probabilities - 1) <= atol`` in this function.
    validate_sum : bool, optional
        whether to validate sum=1, by default True.
    raise_error : bool, optional
        raises error when validation fails, by default True.
    message : str, optional
        prints additional message when validation fails, by default "".

    Raises
    ------
    ValueError
        some elements of prob_dist are negative numbers.
    ValueError
        the sum of prob_dist is not 1.
    """
    if eps == None:
        eps = 1e-8

    # whether each probability is a positive number.
    for index, prob in enumerate(prob_dist):
        # if prob < 0:
        if prob < 0 and not np.isclose(prob, 0, atol=eps, rtol=0.0):
            if raise_error:
                raise ValueError(
                    f"({message}) each probability must be a non-negative number. there is {prob} in a probability distribution({index})"
                )
            else:
                print(
                    f"{start_red}Warning!{end_color} ({message}) each probability must be a non-negative number. there is {prob} in a probability distribution({index})"
                )

    # whether the sum of probabilities equals 1.
    if validate_sum is True:
        sum_p = np.sum(prob_dist)
        if not np.isclose(sum_p, 1.0, atol=eps, rtol=0.0):
            if raise_error:
                raise ValueError(
                    f"({message}) the sum of prob_dist must be 1. the sum of prob_dist={sum_p}"
                )
            else:
                print(
                    f"{start_red}Warning!{end_color} ({message}) the sum of prob_dist must be 1. the sum of prob_dist={sum_p}"
                )
