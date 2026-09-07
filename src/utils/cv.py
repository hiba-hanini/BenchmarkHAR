from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from src.utils.reproducibility import DEFAULT_SEED

def get_cv_splitter(n_subjects: int, cv_config: dict, seed: int = DEFAULT_SEED):
    """
    Determines the cross-validation strategy based on the number of subjects.

    - LeaveOneGroupOut (LOSO) if n_subjects is less than or equal to loso_threshold.
    - GroupShuffleSplit otherwise.

    Args:
        n_subjects: The total number of unique subjects in the dataset.
        cv_config: A dictionary with CV parameters like 'loso_threshold', 
                   'n_splits', and 'test_size'.

    Returns:
        A tuple containing:
        - splitter: The scikit-learn splitter instance.
        - cv_name: The name of the strategy ('LOSO' or 'GroupShuffleSplit').
    """
    loso_threshold = cv_config.get("loso_threshold", 10)
    use_loso = n_subjects <= loso_threshold

    if use_loso:
        splitter = LeaveOneGroupOut()
        cv_name = "LOSO"
    else:
        n_splits = cv_config.get("n_splits", 5)
        test_size = cv_config.get("test_size", 0.2)
        splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
        cv_name = "GroupShuffleSplit"
    
    return splitter, cv_name
