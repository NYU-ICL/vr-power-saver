# Invoker: v0.0.1
# DO NOT MANUALLY EDIT THIS FILE.
#
# This script was generated with invoker.
# To regenerate file, run `invoker rebuild`.
# Date:	01/12/2022
# Hash:	483f9a0695c1159eb6f3061ad4d3e28e
import argparse
import copy
import cProfile as profile
import importlib
import json
import logging
import logging.config
import pstats
import re
from pathlib import Path


def initialize_logger(fname):
    logfile_root = Path("./logs")
    logfile_root.mkdir(exist_ok=True, parents=True)
    logfile_path = logfile_root / f"{fname}.log"
    do_rollover = True if logfile_path.exists() else False
    logger_dict = {
        "version": 1,
        "formatters": {
            "verbose": {
                "format": "%(asctime)s,%(msecs)d [%(levelname)-8s] %(filename)s:%(lineno)d.%(funcName)s() %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "pretty": {
                "format": "%(asctime)s [%(levelname)-8s] %(filename)s:%(lineno)d.%(funcName)s() %(message)s",
                "datefmt": "%H:%M:%S",
                "class": "invoker.InvokerFormatter",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "pretty",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "verbose",
                "filename": logfile_path,
                "maxBytes": 1048576,  # 1MB
                "backupCount": 20,
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"]
        }
    }
    logging.config.dictConfig(logger_dict)
    if do_rollover:
        logging.getLogger("root").handlers[1].doRollover()


class InvokerFormatter(logging.Formatter):
    LVL2COLOR = {
        logging.DEBUG: "\x1b[38m", #  grey
        logging.INFO: "\x1b[36m", #  blue
        logging.WARNING: "\x1b[33m", #  yellow
        logging.ERROR: "\x1b[31m", #  red
        logging.CRITICAL: "\x1b[31;1m", #  bold_red
    }
    RESET = "\x1b[0m"

    def format(self, record):
        out = super().format(record)
        color = self.LVL2COLOR.get(record.levelno)
        return color + logging.Formatter.format(self, record) + self.RESET


class Module:
    def __init__(self, inp_args=None):
        # Build Config
        if inp_args is None:
            conf = self.build_config(self.args())
        else:
            conf = self.build_config(inp_args)
        self.opt = _deserialize_config(conf)
        self.initialize()

    @classmethod
    def args(cls):
        return {}

    @classmethod
    def build_config(cls, args):
        return args

    def initialize(self):
        pass


class Script:
    def __init__(self, inp_args=None):
        self.inp_args = inp_args
        # Initialize logger
        initialize_logger(_to_underscore_case(type(self).__name__))

        # Parse Arguments
        parser = _build_argparser(self.args())
        for module, module_mode in self.modules().items():
            cls = importlib.import_module(module).get_class(module_mode)
            parser = _build_argparser(cls.args(), module, parser)
        self.all_args = vars(parser.parse_args(self.inp_args))

    def initialize(self):
        # Build Config
        conf = self.build_config(self.all_args.copy())
        module_conf = {}
        for module, module_mode in self.modules().items():
            cls = importlib.import_module(module).get_class(module_mode)
            module_args = {
                k.split(".")[1]: v
                for k, v in self.all_args.items()
                if len(k.split(".")) == 2 and k.split(".")[0] == module
            }
            cls_inst = cls(module_args)
            setattr(self, module, cls_inst)
            module_conf[module] = _serialize_opt(cls_inst.opt)
        conf.update(module_conf)

        # Save Config
        if "path" in conf:
            save_root = Path(conf["path"])
            save_root.mkdir(parents=True, exist_ok=True)
            json.dump(
                {
                    "modules": self.modules(),
                    "config": conf,
                },
                open(save_root / "conf.json", "w"))

        # Deserialize Options
        self.opt = _deserialize_config(conf)
        return self

    @classmethod
    def args(cls):
        return {}

    @classmethod
    def modules(cls):
        return {}

    @classmethod
    def build_config(cls, args):
        return args

    def run(self):
        pass

    def profile(self, top=10):
        prof = profile.Profile()
        prof.enable()
        self.run()
        prof.disable()
        stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
        stats.print_stats(top)


class Workflow:
    def __init__(self):
        parser = _build_argparser(self.args())
        all_args = vars(parser.parse_args())
        self.arg_dict = self.build_script_args(all_args)

    @classmethod
    def args(cls):
        return {}

    @classmethod
    def scripts(cls):
        return []

    @classmethod
    def build_script_args(cls, args):
        arg_dict = {}
        for script in cls.scripts():
            arg_dict[script] = {}
        return arg_dict

    @classmethod
    def _generate_arg_list(cls, arg_dict):
        out = []
        for k, v in arg_dict.items():
            out.append(f"--{k}")
            if type(v) == list:
                for item in v:
                    out.append(str(item))
            else:
                out.append(str(v))
        return out

    def run(self):
        for script in self.scripts():
            module = importlib.import_module(script)
            cls = getattr(module, _to_camel_case(script))
            arg_list = self._generate_arg_list(self.arg_dict[script])
            cls_inst = cls(arg_list).initialize()
            cls_inst.run()

    def profile(self, top=10):
        prof = profile.Profile()
        prof.enable()
        self.run()
        prof.disable()
        stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
        stats.print_stats(top)

def _build_argparser(default_args, key_prefix=None, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    def _build_key(kname):
        return f"--{kname}" if key_prefix is None else f"--{key_prefix}.{kname}"
    for k, v in default_args.items():
        try:
            if type(v) == list:
                parser.add_argument(
                    _build_key(k),
                    type=type(v[0]) if len(v) > 0 else str,
                    nargs="+",
                    default=v
                )
            elif type(v) == bool:
                parser.add_argument(
                    _build_key(k),
                    action="store_true" if not v else "store_false",
                )
            else:
                parser.add_argument(
                    _build_key(k),
                    type=type(v),
                    default=v
                )
        except argparse.ArgumentError:
            logging.warn("Script defaults over-riding module arg %s.", k)
            pass
    return parser


def _serialize_opt(opt):
    out = vars(copy.deepcopy(opt))
    for k, v in out.items():
        if isinstance(v, argparse.Namespace):
            out[k] = _serialize_opt(v)
        else:
            out[k] = v
    return out


def _deserialize_config(config):
    opt = argparse.Namespace()
    for k, v in config.items():
        if isinstance(v, dict):
            setattr(opt, k, _deserialize_config(v))
        else:
            setattr(opt, k, v)
    return opt


def _to_camel_case(string):
    return "".join([token.capitalize() for token in string.split("_")])


def _to_underscore_case(string):
    return "_".join([token.lower() for token in re.findall("[A-Z][^A-Z]*", string)])
