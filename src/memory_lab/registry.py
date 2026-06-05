from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_lab.models.base import MemoryModel
    from memory_lab.renderers.base import ContextRenderer
else:
    MemoryModel = Any
    ContextRenderer = Any


ModelFactory = Callable[[], MemoryModel]
RendererFactory = Callable[[], ContextRenderer]

MODEL_ENTRY_POINT_GROUP = "memory_lab.models"
RENDERER_ENTRY_POINT_GROUP = "memory_lab.renderers"


@dataclass(frozen=True)
class ModelRegistration:
    name: str
    factory: ModelFactory
    default_renderer: str = "research_brief"


@dataclass(frozen=True)
class RendererRegistration:
    name: str
    factory: RendererFactory


_MODEL_REGISTRY: dict[str, ModelRegistration] = {}
_RENDERER_REGISTRY: dict[str, RendererRegistration] = {}
_LOADED_MODEL_ENTRY_POINT_GROUPS: set[str] = set()
_LOADED_RENDERER_ENTRY_POINT_GROUPS: set[str] = set()
_MODEL_ENTRY_POINT_ERRORS: dict[str, Exception] = {}
_RENDERER_ENTRY_POINT_ERRORS: dict[str, Exception] = {}


def register_model(
    name_or_factory: str | ModelFactory | None = None,
    factory: ModelFactory | None = None,
    *,
    default_renderer: str = "research_brief",
    replace: bool = False,
) -> Any:
    """Register a memory model factory.

    Supported forms:

    ```python
    register_model("mem0_lite", Mem0LiteMemory, default_renderer="compact_prompt")

    @register_model("mem0_lite", default_renderer="compact_prompt")
    class Mem0LiteMemory(BaseMemoryModel):
        ...
    ```
    """

    if isinstance(name_or_factory, str):
        name = name_or_factory
        if factory is not None:
            _register_model(name, factory, default_renderer, replace=replace)
            return factory

        def decorator(decorated_factory: ModelFactory) -> ModelFactory:
            _register_model(name, decorated_factory, default_renderer, replace=replace)
            return decorated_factory

        return decorator

    if name_or_factory is not None:
        if factory is not None:
            raise TypeError("Pass either a model factory or a name plus factory, not both")
        model_factory = name_or_factory
        name = _name_from_factory(model_factory, "memory model")
        _register_model(name, model_factory, default_renderer, replace=replace)
        return model_factory

    if factory is not None:
        name = _name_from_factory(factory, "memory model")
        _register_model(name, factory, default_renderer, replace=replace)
        return factory

    def decorator(decorated_factory: ModelFactory) -> ModelFactory:
        name = _name_from_factory(decorated_factory, "memory model")
        _register_model(name, decorated_factory, default_renderer, replace=replace)
        return decorated_factory

    return decorator


def register_renderer(
    name_or_factory: str | RendererFactory | None = None,
    factory: RendererFactory | None = None,
    *,
    replace: bool = False,
) -> Any:
    """Register a context renderer factory."""

    if isinstance(name_or_factory, str):
        name = name_or_factory
        if factory is not None:
            _register_renderer(name, factory, replace=replace)
            return factory

        def decorator(decorated_factory: RendererFactory) -> RendererFactory:
            _register_renderer(name, decorated_factory, replace=replace)
            return decorated_factory

        return decorator

    if name_or_factory is not None:
        if factory is not None:
            raise TypeError("Pass either a renderer factory or a name plus factory, not both")
        renderer_factory = name_or_factory
        name = _name_from_factory(renderer_factory, "context renderer")
        _register_renderer(name, renderer_factory, replace=replace)
        return renderer_factory

    if factory is not None:
        name = _name_from_factory(factory, "context renderer")
        _register_renderer(name, factory, replace=replace)
        return factory

    def decorator(decorated_factory: RendererFactory) -> RendererFactory:
        name = _name_from_factory(decorated_factory, "context renderer")
        _register_renderer(name, decorated_factory, replace=replace)
        return decorated_factory

    return decorator


def create_model(name: str) -> MemoryModel:
    model_name = _normalize_name(name, "memory model")
    registration = _MODEL_REGISTRY.get(model_name)
    if registration is None:
        load_model_entry_points()
        registration = _MODEL_REGISTRY.get(model_name)
    if registration is None:
        error = _MODEL_ENTRY_POINT_ERRORS.get(model_name)
        if error is not None:
            raise RuntimeError(f"Failed to load memory model plugin {model_name!r}") from error
        known = ", ".join(list_models(include_plugins=False))
        raise ValueError(f"Unknown memory model {model_name!r}. Known models: {known}")
    return registration.factory()


def create_renderer(name: str) -> ContextRenderer:
    renderer_name = _normalize_name(name, "context renderer")
    registration = _RENDERER_REGISTRY.get(renderer_name)
    if registration is None:
        load_renderer_entry_points()
        registration = _RENDERER_REGISTRY.get(renderer_name)
    if registration is None:
        error = _RENDERER_ENTRY_POINT_ERRORS.get(renderer_name)
        if error is not None:
            raise RuntimeError(f"Failed to load renderer plugin {renderer_name!r}") from error
        known = ", ".join(list_renderers(include_plugins=False))
        raise ValueError(f"Unknown renderer {renderer_name!r}. Known renderers: {known}")
    return registration.factory()


def default_renderer_for(model_name: str, fallback: str = "research_brief") -> str:
    normalized_name = _normalize_name(model_name, "memory model")
    registration = _MODEL_REGISTRY.get(normalized_name)
    if registration is None:
        return fallback
    return registration.default_renderer


def list_models(*, include_plugins: bool = True) -> tuple[str, ...]:
    if include_plugins:
        load_model_entry_points()
    return tuple(sorted(_MODEL_REGISTRY))


def list_renderers(*, include_plugins: bool = True) -> tuple[str, ...]:
    if include_plugins:
        load_renderer_entry_points()
    return tuple(sorted(_RENDERER_REGISTRY))


def load_model_entry_points(group: str = MODEL_ENTRY_POINT_GROUP) -> None:
    if group in _LOADED_MODEL_ENTRY_POINT_GROUPS:
        return
    _LOADED_MODEL_ENTRY_POINT_GROUPS.add(group)
    for entry_point in _entry_points_for(group):
        if entry_point.name in _MODEL_REGISTRY:
            continue
        try:
            factory = entry_point.load()
        except Exception as exc:  # pragma: no cover - depends on installed plugins
            _MODEL_ENTRY_POINT_ERRORS[entry_point.name] = exc
            continue
        default_renderer = str(getattr(factory, "default_renderer", "research_brief"))
        _register_model(entry_point.name, factory, default_renderer, replace=False)


def load_renderer_entry_points(group: str = RENDERER_ENTRY_POINT_GROUP) -> None:
    if group in _LOADED_RENDERER_ENTRY_POINT_GROUPS:
        return
    _LOADED_RENDERER_ENTRY_POINT_GROUPS.add(group)
    for entry_point in _entry_points_for(group):
        if entry_point.name in _RENDERER_REGISTRY:
            continue
        try:
            factory = entry_point.load()
        except Exception as exc:  # pragma: no cover - depends on installed plugins
            _RENDERER_ENTRY_POINT_ERRORS[entry_point.name] = exc
            continue
        _register_renderer(entry_point.name, factory, replace=False)


def _register_model(
    name: str,
    factory: ModelFactory,
    default_renderer: str,
    *,
    replace: bool,
) -> None:
    model_name = _normalize_name(name, "memory model")
    existing = _MODEL_REGISTRY.get(model_name)
    if existing is not None and not replace and existing.factory is not factory:
        raise ValueError(f"Memory model {model_name!r} is already registered")
    renderer_name = _normalize_name(default_renderer, "default renderer")
    _MODEL_REGISTRY[model_name] = ModelRegistration(
        name=model_name,
        factory=factory,
        default_renderer=renderer_name,
    )


def _register_renderer(
    name: str,
    factory: RendererFactory,
    *,
    replace: bool,
) -> None:
    renderer_name = _normalize_name(name, "context renderer")
    existing = _RENDERER_REGISTRY.get(renderer_name)
    if existing is not None and not replace and existing.factory is not factory:
        raise ValueError(f"Renderer {renderer_name!r} is already registered")
    _RENDERER_REGISTRY[renderer_name] = RendererRegistration(
        name=renderer_name,
        factory=factory,
    )


def _name_from_factory(factory: Callable[..., object], label: str) -> str:
    raw_name = getattr(factory, "name", None)
    return _normalize_name(raw_name, label)


def _normalize_name(value: object, label: str) -> str:
    if value is None:
        raise ValueError(f"A registered {label} must have a name")
    name = str(value).strip()
    if not name:
        raise ValueError(f"A registered {label} must have a non-empty name")
    return name


def _entry_points_for(group: str) -> tuple[EntryPoint, ...]:
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return tuple(discovered.select(group=group))
    return tuple(discovered.get(group, ()))  # type: ignore[attr-defined]


__all__ = [
    "MODEL_ENTRY_POINT_GROUP",
    "ModelRegistration",
    "RENDERER_ENTRY_POINT_GROUP",
    "RendererRegistration",
    "create_model",
    "create_renderer",
    "default_renderer_for",
    "list_models",
    "list_renderers",
    "load_model_entry_points",
    "load_renderer_entry_points",
    "register_model",
    "register_renderer",
]
