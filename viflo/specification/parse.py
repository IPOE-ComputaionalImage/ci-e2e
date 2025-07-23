import abc
import json
import logging
import runpy
from pathlib import Path
import typing

import yaml

from .template import Template

__all__ = [
    'parse_spec_file',

    'FromSpecification',
    'SpecificationError',
]
logger = logging.getLogger(__name__)


def _filter_special(global_dict):
    return {k: v for k, v in global_dict.items() if not k.startswith('_')}


def _parse_py(file_path: Path) -> Template:
    results = runpy.run_path(str(file_path))
    globals()['__viflo_artifacts__'] = results  # to avoid GC

    template_cls = None
    template = None
    for item in results.values():
        if isinstance(item, Template):
            if template is None:  # at most one template instance
                template = item
            else:
                raise SpecificationError(f'Multiple template instances found in {file_path}')
        if isinstance(item, type) and issubclass(item, Template) and item is not Template:
            if template_cls is None:  # at most one template class
                template_cls = item
            else:
                raise SpecificationError(f'Multiple template classes found in {file_path}')

    if template is not None:  # use the found template instance
        if template_cls is not None and template_cls is not template.__class__:
            raise SpecificationError(f'A {template.__class__.__name__} instance '
                                     f'and {template_cls.__name__} class found in {file_path}')
        return template

    if template_cls is None:  # construct a template from global variables
        template = Template(**_filter_special(results))
    else:  # construct a template from the found template class
        template = template_cls()

    logger.info(f'Specification resolved ({template.__class__.__name__})')
    return template


def _parse_json(file_path: Path) -> Template:
    with file_path.open('r') as f:
        data = json.load(f)
    return Template(**data)


def _parse_yaml(file_path: Path) -> Template:
    with file_path.open('r') as f:
        data = yaml.safe_load(f)
    return Template(**data)


class SpecificationError(RuntimeError):
    pass


def parse_spec_file(file_path: str | Path) -> Template:
    file_path = Path(file_path)
    logger.info(f'Resolving specification from {file_path.suffix} file')
    if file_path.suffix == '.py':
        return _parse_py(file_path)
    elif file_path.suffix == '.json':
        return _parse_json(file_path)
    elif file_path.suffix == '.yaml' or file_path.suffix == '.yml':
        return _parse_yaml(file_path)


class FromSpecification(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_specification(cls, spec: Template) -> typing.Self:
        pass
