# stdlib imports --------------------------------------------------- #
import argparse
import dataclasses
import json
import functools
import logging
import pathlib
import time
import types
import typing
import uuid
from typing import Any, Literal

# 3rd-party imports necessary for processing ----------------------- #
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pynwb
import upath
import zarr
import xarray as xr

import utils
import models
import glm_utils

# shs

# logging configuration -------------------------------------------- #
# use `logger.info(msg)` instead of `print(msg)` so we get timestamps and origin of log messages
logger = logging.getLogger(
    pathlib.Path(__file__).stem if __name__.endswith("_main__") else __name__
    # multiprocessing gives name '__mp_main__'
)

# general configuration -------------------------------------------- #
matplotlib.rcParams['pdf.fonttype'] = 42
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR) # suppress matplotlib font warnings on linux


# utility functions ------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging_level', type=str, default='INFO')
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--update_packages_from_source', type=int, default=1)
    parser.add_argument('--override_params_json', type=str, default="{}")
    for field in dataclasses.fields(AppParams):
        if field.name in [getattr(action, 'dest') for action in parser._actions]:
            # already added field above
            continue
        logger.debug(f"adding argparse argument {field}")
        kwargs = {}
        if isinstance(field.type, str):
            kwargs = {'type': eval(field.type)} 
        else:
            kwargs = {'type': field.type}
        if kwargs['type'] in (list, tuple):
            logger.debug(f"Cannot correctly parse list-type arguments from App Builder: skipping {field.name}")
        if isinstance(field.type, str) and field.type.startswith('Literal'):
            kwargs['type'] = str
        if isinstance(kwargs['type'], (types.UnionType, typing._UnionGenericAlias)):
            kwargs['type'] = typing.get_args(kwargs['type'])[0]
            logger.debug(f"setting argparse type for union type {field.name!r} ({field.type}) as first component {kwargs['type']!r}")
        parser.add_argument(f'--{field.name}', **kwargs)
    args = parser.parse_args()
    list_args = [k for k,v in vars(args).items() if type(v) in (list, tuple)]
    if list_args:
        raise NotImplementedError(f"Cannot correctly parse list-type arguments from App Builder: remove {list_args} parameter and provide values via `override_params_json` instead")
    logger.info(f"{args=}")
    return args

# processing function ---------------------------------------------- #
# modify the body of this function, but keep the same signature
def process(app_params: "AppParams", inputs_path: str | pathlib.Path, fullmodel_outputs_path: str | pathlib.Path | None = None, test: int = 0) -> None:

    """Process a single session with parameters defined in `params` and save results + params to
    /results.
    
    A test mode should be implemented to allow for quick testing of the capsule (required every time
    a change is made if the capsule is in a pipeline) 
    """
    inputs_path = pathlib.Path(inputs_path)
    logger.info(f"Processing {inputs_path.name}")
    
    input_dict = np.load(inputs_path, allow_pickle=True)
    run_params = input_dict['run_params'].item()
    fit = input_dict['fit'].item()
    design_matrix_dict = input_dict['design_matrix'].item()

    design_matrix = xr.Dataset({
                        "data": (["rows", "columns"], design_matrix_dict["data"]),
                        "weights": (["columns"], design_matrix_dict["weights"]),
                        "timestamps": (["rows"], design_matrix_dict["timestamps"])
                        })

    session_id = run_params["session_id"]
    model_params = glm_utils.RunParams(session_id = session_id)
    model_params.update_multiple_metrics(app_params)

    if fullmodel_outputs_path:
        logger.info(f"Re-using regularization parameters from {fullmodel_outputs_path.name}")
        fullmodel_outputs_path = pathlib.Path(fullmodel_outputs_path)
        fullmodel_dict =  np.load(fullmodel_outputs_path, allow_pickle=True)

        model_params.update_multiple_metrics({'fullmodel_fitted': True, 
                'cell_regularization_nested': fullmodel_dict['cell_regularization_nested'],
                'cell_regularization': fullmodel_dict['cell_regularization'],
                'cell_rank_nested': fullmodel_dict['cell_rank_nested'],
                'cell_rank': fullmodel_dict['cell_rank'],
                'cell_L1_ratio_nested': fullmodel_dict['cell_L1_ratio_nested'],
                'cell_L1_ratio': fullmodel_dict['cell_L1_ratio']})
        # incorporate params

    # get all parameters
    model_params.validate_params()
    run_params = run_params | model_params.get_params()

    fit = glm_utils.optimize_model(fit, design_matrix, run_params)
    fit = glm_utils.evaluate_model(fit, design_matrix, run_params)
    fit['spike_count_arr'].pop('spike_counts', None)

    # Save data to files in /results
    # If the same name is used across parallel runs of this capsule in a pipeline, a name clash will
    # occur and the pipeline will fail, so use session_id as filename prefix:
    #   /results/<sessionId>.suffix
    results = {
        'fit': fit,
        'params': run_params,
    }
    output_path = pathlib.Path(f"/results/outputs/{session_id}_{run_params['model_label']}_outputs.npz")
    # /outputs/ avoids name clash due to multiple logs dirs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing results to {output_path}")
    np.savez(output_path, **results)
    if not output_path.exists():
        raise AssertionError(f"{output_path} should exist after writing")

# define run params here ------------------------------------------- #

# The `Params` class is used to store parameters for the run, for passing to the processing function.
# @property fields (like `bins` below) are computed from other parameters on-demand as required:
# this way, we can separate the parameters dumped to json from larger arrays etc. required for
# processing.

# - if needed, we can get parameters from the command line (like `nUnitSamples` below) and pass them
#   to the dataclass (see `main()` below)

# this is an example from Sam's processing code, replace with your own parameters as needed:
@dataclasses.dataclass
class AppParams:

    method: str = 'ridge_regression',  # ['ridge_regression', 'lasso_regression', ...]

    no_nested_CV: bool = False,
    optimize_on: float = 0.3,
    n_outer_folds: int = 5,
    n_inner_folds: int = 5,
    optimize_penalty_by_cell: bool = False,
    optimize_penalty_by_area: bool = False,
    optimize_penalty_by_firing_rate: bool = False,
    optimize_penalty_by_best_units: bool = False, # TO DO
    use_fixed_penalty: bool =  False,
    num_rate_clusters: int = 5,

    # RIDGE + ELASTIC NET
    L2_grid_type: str ='log',
    L2_grid_range: list = [1, 2**12],
    L2_grid_num: int = 13,
    L2_fixed_lambda: float = None,

    # LASSO
    L1_grid_type: str = 'log',
    L1_grid_range: list = [10**-6, 10**-2],
    L1_grid_num: int = 13,
    L1_fixed_lambda: float = None,

    # ELASTIC NET
    L1_ratio_grid_type: str = 'log',
    L1_ratio_grid_range: list = [10**-6, 10**-1],
    L1_ratio_grid_num: int = 9,
    L1_ratio_fixed: float = None,

    # RRR
    rank_grid_num: int = 10,
    rank_fixed: float = None,

    @property
    def bins(self) -> npt.NDArray[np.float64]:
        return np.arange(self.binStart, self.windowDur+self.binSize, self.binSize)

    @property
    def nBins(self) -> int:
        return self.bins.size - 1
    
    def to_dict(self) -> dict[str, Any]:
        """dict of field name: value pairs, including values from property getters"""
        return dataclasses.asdict(self) | {k: getattr(self, k) for k in dir(self.__class__) if isinstance(getattr(self.__class__, k), property)}

    def to_json(self, **dumps_kwargs) -> str:
        """json string of field name: value pairs, excluding values from property getters (which may be large)"""
        return json.dumps(dataclasses.asdict(self), **dumps_kwargs)

    def write_json(self, path: str | upath.UPath = '/results/params.json') -> None:
        path = upath.UPath(path)
        logger.info(f"Writing params to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=2))

# ------------------------------------------------------------------ #


def main():
    t0 = time.time()
    
    utils.setup_logging()

    # get arguments passed from command line (or "AppBuilder" interface):
    args = parse_args()
    logger.setLevel(args.logging_level)

    # if any of the parameters required for processing are passed as command line arguments, we can
    # get a new params object with these values in place of the defaults:

    app_params = {}
    for field in dataclasses.fields(AppParams):
        if (val := getattr(args, field.name, None)) is not None:
            app_params[field.name] = val
    
    override_params = json.loads(args.override_params_json)
    if override_params:
        for k, v in override_params.items():
            if k in app_params:
                logger.info(f"Overriding value of {k!r} from command line arg with value specified in `override_params_json`")
            app_params[k] = v
            
    # if test mode is on, we process .npz files attached to the capsule in /code,
    # otherwise, process all .npz files discovered in /data
    if args.test:
        data_path = pathlib.Path('/code')
    else: 
        data_path = utils.get_data_root()
    input_dict_paths = tuple(data_path.rglob('*_inputs.npz'))
    logger.info(f"Found {len(input_dict_paths)} inputs .npz file(s)")

    fullmodel_outputs_paths = tuple(data_path.rglob('*_fullmodel_outputs.npz'))
    logger.info(f"Found {len(fullmodel_outputs_paths)} full model outputs .npz file(s)")

    # run processing function for each .npz file, with test mode implemented:
    for input_dict_path in input_dict_paths:
        session_id = '_'.join(input_dict_path.stem.split('_')[:2])
        matching_outputs = tuple(
            f for f in fullmodel_outputs_paths if f.stem.startswith(session_id)
        )
        if len(matching_outputs) > 1:
            raise AssertionError(f"Multiple files found for outputs of full model for {session_id}: something has likely gone wrong in data connections in pipeline")
        fullmodel_outputs_path = matching_outputs[0] if matching_outputs else None
        try:
            # may need two sets of params (one for model params, one for configuring how model is run, e.g. parallelized)
            process(inputs_path=input_dict_path, fullmodel_outputs_path=fullmodel_outputs_path, app_params = app_params, test=args.test)
        except Exception as e:
            logger.exception(f'{input_dict_path.stem} | Failed:')
        else:
            logger.info(f'{input_dict_path.stem} | Completed')

        if args.test:
            logger.info("Test mode: exiting after first session")
            break
    utils.ensure_nonempty_results_dirs(['/results', '/results/outputs'])
    logger.info(f"Time elapsed: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()


# To fix, if its a reduced model, it shouldnt run until it finds the fullmodel
