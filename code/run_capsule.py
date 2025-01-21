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

import utils

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
    for field in dataclasses.fields(Params):
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
def process(inputs_path: str | pathlib.Path, full_model_outputs_path: str | pathlib.Path | None = None, test: int = 0) -> None:
    """Process a single session with parameters defined in `params` and save results + params to
    /results.
    
    A test mode should be implemented to allow for quick testing of the capsule (required every time
    a change is made if the capsule is in a pipeline) 
    """
    inputs_path = pathlib.Path(inputs_path)
    logger.info(f"Processing {inputs_path.name}")
    
    npz = np.load(inputs_path, allow_pickle=True)
    params = npz['params'].item()

    if full_model_outputs_path:
        logger.info(f"Re-using regularization parameters from {full_model_outputs_path.name}")
        # incorporate params

    # run GLM...

    # Save data to files in /results
    # If the same name is used across parallel runs of this capsule in a pipeline, a name clash will
    # occur and the pipeline will fail, so use session_id as filename prefix:
    #   /results/<sessionId>.suffix
    results = {
        'x': np.full((5,5), 1.2), 
        'y': np.full((5,5), 1.2), 
        'fit': np.full((5,5), 1.2),
        'params': params,
    }
    output_path = pathlib.Path(f"/results/outputs/{params['session_id']}_{params['model_name']}_outputs.npz")
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
class Params:
    
    nUnitSamples: int = 20
    unitSampleSize: int = 20
    windowDur: float = 1
    binSize: float = 1
    nShuffles: int | str = 100
    binStart: int = -windowDur
    n_units: list = dataclasses.field(default_factory=lambda: [5, 10, 20, 40, 60, 'all'])
    decoder_type: str | Literal['linearSVC', 'LDA', 'RandomForest', 'LogisticRegression'] = 'LogisticRegression'

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
    
    #utils.setup_logging()

    # get arguments passed from command line (or "AppBuilder" interface):
    args = parse_args()
    logger.setLevel(args.logging_level)

    # if any of the parameters required for processing are passed as command line arguments, we can
    # get a new params object with these values in place of the defaults:

    params = {}
    for field in dataclasses.fields(Params):
        if (val := getattr(args, field.name, None)) is not None:
            params[field.name] = val
    
    override_params = json.loads(args.override_params_json)
    if override_params:
        for k, v in override_params.items():
            if k in params:
                logger.info(f"Overriding value of {k!r} from command line arg with value specified in `override_params_json`")
            params[k] = v
            
    # if test mode is on, we process .npz files attached to the capsule in /code,
    # otherwise, process all .npz files discovered in /data
    if args.test:
        data_path = pathlib.Path('/code')
    else: 
        data_path = utils.get_data_root()
    npz_paths = tuple(data_path.rglob('*_inputs.npz'))
    logger.info(f"Found {len(npz_paths)} .npz paths available for use")
    
    full_model_outputs_paths = tuple(data_path.rglob('*_full_model_outputs.npz'))
    logger.info(f"Found {len(full_model_outputs_paths)} full model outputs .npz paths available for use")
    if len(full_model_outputs_paths) > 1:
        raise NotImplementedError(f"Multiple files found for outputs of full model: implement matching of output files to input files")
    full_model_outputs_path = full_model_outputs_paths[0] if full_model_outputs_paths else None
    
    # run processing function for each .npz file, with test mode implemented:
    for npz_path in npz_paths:
        try:
            # may need two sets of params (one for model params, one for configuring how model is run, e.g. parallelized)
            process(inputs_path=npz_path, full_model_outputs_path=full_model_outputs_path, test=args.test)
        except Exception as e:
            logger.exception(f'{npz_path.stem} | Failed:')
        else:
            logger.info(f'{npz_path.stem} | Completed')

        if args.test:
            logger.info("Test mode: exiting after first session")
            break
    utils.ensure_nonempty_results_dirs(['/results', '/results/outputs'])
    logger.info(f"Time elapsed: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
