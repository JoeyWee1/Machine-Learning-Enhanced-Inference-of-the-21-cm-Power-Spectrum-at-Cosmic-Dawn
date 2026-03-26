import numpy as np
from pathlib import Path

def unpack_simulations(simulations: list[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Unpacks a list of input simulations into separate arrays for parameters, power spectra, and k values.

    Args:
        simulations (list[dict]): A list of dictionaries, where each dictionary contains the keys "astro_params", "cosmo_params", "power", and "k".
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three numpy arrays:
            - params_array: An array of shape (N, 4) where N is the number of simulations, and each row contains the parameters [L40_xray, fesc10, epsstar, h_fid].
            - power_array: An array of shape (N, K) where K is the number of k values, containing the power spectra for each simulation.
            - k_array: An array of shape (N, K) containing the k values for each simulation.
    '''
    params_list = []
    power_list = []
    k_list = []

    for sim in simulations:
        astro = sim["astro_params"].item()
        cosmo = sim["cosmo_params"].item()
        params_list.append(
            [
                astro["L40_xray"],
                astro["fesc10"],
                astro["epsstar"],
                cosmo["h_fid"],
            ]
        )
        power_list.append(sim["power"])
        k_list.append(sim["k"])

    return np.asarray(params_list), np.asarray(power_list), np.asarray(k_list)


def load_splits(data_dir: Path) -> dict:
    '''
    Loads and splits simulation data from .npz files in the specified directory.

    Args:
        data_dir (Path): The directory containing the .npz files.

    Returns:
        dict: A dictionary containing the training, validation, and test splits for parameters, power spectra, and k values, as well as the corresponding file paths.
    '''
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    num_files = len(files)
    train_files = files[: int(0.8 * num_files)]
    val_files = files[int(0.8 * num_files) : int(0.9 * num_files)]
    test_files = files[int(0.9 * num_files) :]

    def read_many(file_list: list[Path]) -> list[dict]:
        sims = []
        for f in file_list:
            with np.load(f, allow_pickle=True) as d:
                sims.append(dict(d))
        return sims

    train_sims = read_many(train_files)
    val_sims = read_many(val_files)
    test_sims = read_many(test_files)

    raw_params_train, power_train, k_train = unpack_simulations(train_sims)
    raw_params_val, power_val, k_val = unpack_simulations(val_sims)
    raw_params_test, power_test, k_test = unpack_simulations(test_sims)

    return {
        "raw_params_train": raw_params_train,
        "raw_params_val": raw_params_val,
        "raw_params_test": raw_params_test,
        "power_train": power_train,
        "power_val": power_val,
        "power_test": power_test,
        "k_train": k_train,
        "k_val": k_val,
        "k_test": k_test,
        "train_files": [str(p) for p in train_files],
        "val_files": [str(p) for p in val_files],
        "test_files": [str(p) for p in test_files],
    }