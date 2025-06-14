import abc
import runpy
import typing
from pathlib import Path

from .template import Template

__all__ = [
    'parse_spec_file',

    'FromSpecification',
]


def parse_spec_file(file_path: str | Path) -> Template:
    file_path = str(file_path)

    results = runpy.run_path(file_path)

    template_cls = None
    for item in results.values():
        if isinstance(item, type) and issubclass(item, Template) and item is not Template:
            if template_cls is None:
                template_cls = item
            else:
                raise ValueError(f'Multiple template classes found in {file_path}')

    template = template_cls()
    return template


class FromSpecification(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_specification(cls, spec: Template) -> typing.Self:
        pass
