import abc
import runpy
import typing
from pathlib import Path

from .template import Template

__all__ = [
    'parse_spec_file',

    'FromSpecification',
    'SpecificationError',
]

def _filter_special(global_dict):
    return {k: v for k, v in global_dict.items() if not k.startswith('_')}


class SpecificationError(RuntimeError):
    pass


def parse_spec_file(file_path: str | Path) -> Template:
    file_path = str(file_path)

    results = runpy.run_path(file_path)

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
    return template


class FromSpecification(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_specification(cls, spec: Template) -> typing.Self:
        pass
