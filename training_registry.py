import copy
import inspect
import itertools
import logging
import math
from typing import Any, Dict, List, Type, TypeVar, Union, get_args, get_origin

import attrs
import omegaconf as oc

T = TypeVar("T")

oc.OmegaConf.register_new_resolver("sum", lambda *numbers: sum(numbers))
oc.OmegaConf.register_new_resolver("prod", lambda *numbers: math.prod(numbers))
oc.OmegaConf.register_new_resolver("range", lambda *args: oc.ListConfig(list(range(*args))))
oc.OmegaConf.register_new_resolver("len", lambda *arg: len(arg[0]))
oc.OmegaConf.register_new_resolver("concat", lambda *args: oc.ListConfig(list(itertools.chain(*args))))
oc.OmegaConf.register_new_resolver("n_first", lambda a, b: oc.ListConfig(list(a)[:b]))
oc.OmegaConf.register_new_resolver("n_equally_spaced", lambda a, n: a[:: (len(a) // n)])


logger = logging.getLogger()


def str_target_serializer(obj, k, v):
    if k is not None and k.name == "_target":
        mod = inspect.getmodule(v)
        assert mod is not None
        name = v.__name__
        v = f"{mod.__name__}.{name}"
    return v


class Registry:
    registries: Dict[str, "Registry"] = {}

    def __init__(self, name: str):
        self._name = name
        self._reg = {}

        type(self).registries[name] = self

    def register(self, obj):
        short_name = obj.__name__
        mod_name = inspect.getmodule(obj).__name__
        self._reg[f"{mod_name}.{short_name}"] = obj

        return obj

    def __contains__(self, item):
        return item in self._reg

    def __getitem__(self, item):
        return self._reg[item]

    def __iter__(self):
        return self._reg.items().__iter__()

    @classmethod
    def get_registry(cls, name: str) -> "Registry":
        assert name in cls.registries
        return cls.registries[name]


@attrs.define
class BaseConfig:
    @attrs.define
    class TypedContainer:
        def asdict(self) -> Dict[str, Any]:
            return attrs.asdict(self)

        @classmethod
        def fromdict(cls: Type[T], src: Dict[str, Any]) -> T:
            src = copy.deepcopy(src)
            fields_dict = attrs.fields_dict(cls)
            for field in fields_dict.values():
                if isinstance(field.type, type) and issubclass(field.type, BaseConfig.TypedContainer):
                    src[field.name] = field.type(**src[field.name])
            return cls(**src)

    def asdict(self) -> Dict[str, Any]:
        return attrs.asdict(self, value_serializer=str_target_serializer)

    def get_target(self) -> Type[T]:
        assert hasattr(self, "_target")
        assert isinstance(self._target, type), f"_target attr must be `type`, but is of type {type(self._target)}"
        return self._target

    @classmethod
    def fromdict(cls: Type[T], src: Dict[str, Any]) -> T:
        src = copy.deepcopy(src)
        fields_dict = attrs.fields_dict(cls)
        for field in fields_dict.values():
            if isinstance(field.type, type) and issubclass(field.type, BaseConfig):
                src[field.name] = get_targeted_config_from_dict(singleton, src[field.name])
            elif get_origin(field.type) is Union and all(
                issubclass(maybe_target, BaseConfig) for maybe_target in get_args(field.type)
            ):
                src[field.name] = get_targeted_config_from_dict(singleton, src[field.name])
            elif get_origin(field.type) is dict and issubclass(get_args(field.type)[1], BaseConfig.TypedContainer):
                _, sub_target = get_args(field.type)
                dict_to_init_from: Dict[str, Dict[str, Any]] = src[field.name]
                src[field.name] = {k: sub_target(**v) for k, v in dict_to_init_from.items()}
            elif get_origin(field.type) is list and issubclass(get_args(field.type)[0], BaseConfig.TypedContainer):
                (sub_target,) = get_args(field.type)
                dict_to_init_from: List[Dict[str, Any]] = src[field.name]
                src[field.name] = [sub_target(**v) for v in dict_to_init_from]
            elif isinstance(field.type, type) and issubclass(field.type, BaseConfig.TypedContainer):
                src[field.name] = field.type(**src[field.name])
            else:
                continue

        src.pop("_target")
        return cls(**src)


ConfigImpl = TypeVar("ConfigImpl", bound=BaseConfig)


def get_targeted_config_from_dict(registry: Registry, config: Dict[str, Any]) -> ConfigImpl:
    assert "_target" in config

    target: str = config["_target"]

    # reverse compatibility change to be able to work with models
    # trained on junk version of this pipeline
    target = target.replace("humans.", "reconstruction.neuralbody.", 1)
    assert target in registry

    reg_type = registry[target]

    assert hasattr(reg_type, "ConfigType")
    assert isinstance(reg_type.ConfigType, type)

    assert issubclass(reg_type.ConfigType, BaseConfig)
    config_type: Type[BaseConfig] = reg_type.ConfigType

    return config_type.fromdict(config)


def init_from_typed_config(config: ConfigImpl) -> T:
    if isinstance(config, BaseConfig):
        target = config.get_target()
        return target(config)


def init_from_config(registry: Registry, config: Dict[str, Any]) -> T:
    try:
        logger.info("trying to init from BaseConfig")
        typed_config = get_targeted_config_from_dict(registry, config)
    except AssertionError as e:
        logger.info(f"get_targeted_config_from_dict: {e}")
        typed_config = None
    if typed_config is not None:
        instance = init_from_typed_config(typed_config)
        logger.info("init from BaseConfig suceeded")
        return instance

    try:
        config = copy.deepcopy(config)
        logger.info("trying to init from kwargs")
        assert "_target" in config
        target = registry[config.pop("_target")]
        instance = target(**config)
        logger.info("init from kwargs suceeded")
        return instance
    except (AssertionError, ValueError):
        logger.info("init from kwargs failed")


singleton = Registry("lookup")

