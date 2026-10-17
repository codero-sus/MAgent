# Plugins and hooks

MAgent plugins are directories with a `plugin.json` manifest and a Python entrypoint. The loader can discover and inspect manifests without importing code, then enforce allow/deny/checksum/permission policy before import.

## Scaffold

```bash
magent plugins scaffold ./plugins --name weather
magent plugins inspect ./plugins/weather
magent plugins list --dir ./plugins
magent plugins load --dir ./plugins
```

Generated layout:

```text
plugins/weather/
├── plugin.json
└── plugin.py
```

The generated plugin has one real `weather_hello` tool and one `before_run` hook.

## Manifest

```json
{
  "manifest_version": 1,
  "name": "weather",
  "version": "0.1.0",
  "entrypoint": "plugin.py",
  "summary": "Local weather tools",
  "permissions": ["tools", "hooks"],
  "tools": ["weather_current"],
  "hooks": ["before_run"],
  "enabled": true
}
```

Other fields include author, homepage, license, minimum MAgent version, configuration schema, required environment names, checksum, and extension metadata.

Names and versions are validated. Entrypoints must be relative, use a supported Python suffix, and resolve inside the plugin directory. A minimum version newer than the running MAgent is refused.

## Permissions

Known permissions:

- `tools`;
- `hooks`;
- `filesystem.read`;
- `filesystem.write`;
- `network`;
- `shell`;
- `memory.read`;
- `memory.write`;
- `secrets.read`;
- `config.read`.

Sensitive permissions are `filesystem.write`, `network`, `shell`, and `secrets.read`. Declaring one is not approval:

```bash
magent plugins load --dir ./plugins --allow-sensitive network
```

The loader also accepts plugin-name allow and deny lists. `--deny weather` means the plugin named `weather`; it does not mean a denied permission.

## Checksum

A manifest can pin SHA-256 over its Python files. When present, verification is enabled by default and modified code is refused. `--no-checksum` is an explicit operator override.

No checksum means “not pinned,” not “a generated checksum magically authenticates the publisher.” Distribute expected hashes through a trusted channel.

## Tool export

```python
from magent import tool

@tool(name="weather_current", category="weather", risk="read")
def current(city: str) -> dict[str, object]:
    """Return a current observation.

    Args:
        city: city name
    """
    return {"city": city, "temperature_c": 24}

TOOLS = [current]
```

The loader registers tools only when:

- the manifest grants `tools`;
- a registry was supplied;
- the tool is listed in `manifest.tools` (when the list is non-empty);
- no existing tool conflicts.

An optional module-level `load(registry)` can perform registration for compatible plugin styles.

## Hooks

Lifecycle names:

```text
before_run   after_run    before_tool   after_tool   on_event
on_error     build_prompt filter_output startup      shutdown
```

Decorator:

```python
from magent.plugins import hook

@hook("before_run", plugin="weather", priority=50)
def validate_location(payload):
    if payload.get("forbidden"):
        return False
    return None
```

Hook behavior:

- lower priorities run first;
- equal priorities preserve insertion order;
- `once=True` fires once;
- observer return values are ignored;
- transform hooks (`before_tool`, `filter_output`, `build_prompt`) thread return values;
- veto hooks (`before_run`, `before_tool`) stop on `False` or an object whose `allow` is false;
- exceptions are recorded and the chain continues, unless strict mode is enabled.

## Python loader

```python
from magent.plugins import HookRegistry, PluginLoader
from magent.tools import default_registry

registry = default_registry()
hooks = HookRegistry()
loader = PluginLoader(
    ["./plugins"],
    registry=registry,
    hooks=hooks,
    allow_sensitive=["network"],
    verify_checksums=True,
)
report = loader.load_all()
print(report.render())
```

`with_policy()` returns an independent loader policy while sharing tool/hook registries. This is useful for command-local extra directories without mutating application configuration.

## Module identity

Imported modules use `magent_plugin_<name>`. If that name is already loaded from a different resolved entrypoint, the second plugin is refused. Reusing stale code would bypass the newly inspected file/checksum; replacing it would leave old registered tools alive. Call `unload(name)` explicitly before moving/reloading a plugin.

Unloading removes hooks and module state. Registered tools remain until the registry is rebuilt; the loader says so explicitly.

## Static inspection versus trust

`magent plugins inspect` parses and verifies files without importing the entrypoint. `load` executes Python. Permission declarations help policy and review, but Python imports are not capability-secure. Use a container/VM/restricted account for untrusted plugins.
