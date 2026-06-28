# miles-wheels-rocm

Pre-built **ROCm** wheels for [Miles](https://github.com/radixark/miles) — the AMD
counterpart of [`yueming-yuan/miles-wheels`](https://github.com/yueming-yuan/miles-wheels).

`docker/Dockerfile.rocm` downloads these (via `WHEELS_REPO` / `WHEELS_TAG_ROCM`) and
`pip install`s them instead of compiling Transformer Engine and flash-attn from source.

## Releases

Each tag encodes **(ROCm minor × GPU arch × miles base)**. The `.whl` files are the
release **assets**.

| Tag | ROCm | GPU arch | Wheels |
| --- | --- | --- | --- |
| `rocm700-gfx950-v0.5.14` | 7.0 | gfx950 (MI350/MI355X) | `transformer_engine` 2.8.0, `flash_attn` 2.8.3 (cp310) |

## How they're built

CPU cross-compile inside the `rocm/sgl-dev` base (no GPU needed), the same recipe
`docker/Dockerfile.rocm` used to build inline, but emitting `.whl` via `pip wheel`.
See [`build/in-container-build.sh`](build/in-container-build.sh).
