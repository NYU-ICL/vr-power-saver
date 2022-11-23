# Invoker: v0.0.1
# DO NOT MANUALLY EDIT THIS FILE.
#
# This script was generated with invoker.
# To regenerate file, run `invoker rebuild`.
# Date:	22/11/2022
# Hash:	5bc6446d0237116bef6325cd4920e0ea
import importlib
import importlib.resources
import inspect


# Import all module classes
_PACKAGE_NAME = "".join([token.capitalize() for token in __package__.split("_")])
_CLASSES = {
    name: cls
    for list_of_classes in [
        inspect.getmembers(module, inspect.isclass)
        for module in [
            importlib.import_module(f"{__package__}.{fname.rstrip('.py')}")
            for fname in importlib.resources.contents(__package__)
            if fname.endswith(".py") and fname != "__init__.py"
        ]
    ]
    for name, cls in list_of_classes
    if name.endswith(_PACKAGE_NAME)
}


def get_class(mode):
    cls_name = (
        "".join([token.capitalize() for token in mode.split("_")])
        + _PACKAGE_NAME
    )
    if cls_name not in _CLASSES.keys():
        raise TypeError(f"Invalid {__package__} mode: {cls_name}")
    return _CLASSES[cls_name]
