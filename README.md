# FlagTrain

English | [中文](README_cn.md)

## Introduction

FlagTrain is part of [FlagOS](https://flagos.io/), an open-source system software stack for diverse AI chips. FlagOS connects models, systems, and chips through software, reducing the cost of porting and maintaining models across hardware platforms.

FlagTrain is a library of training operators and compute components implemented in [Triton](https://github.com/triton-lang/triton). It targets mainstream training frameworks and key components, with reusable training computation implementations that can be optimized to accelerate large-model training across multiple hardware backends.

For training framework developers, FlagTrain aims to provide Triton implementations aligned with the relevant interfaces and computational semantics, reducing repeated development and hardware adaptation work. For operator developers, FlagTrain provides a collaborative platform for organizing implementations, tests, and performance optimization around real training workloads.

The project is under active development: documentation and component implementations are added incrementally, and component and hardware compatibility have not been fully validated.

## Installation

FlagTrain does not publish or maintain its own container images. The shared FlagOS
software stack is built by
[flagos-ai/build-infra](https://github.com/flagos-ai/build-infra) from its
[`configs.yaml`](https://github.com/flagos-ai/build-infra/blob/main/configs.yaml).
Use the public Megatron application image (`megatron_training`), which is the
FlagOS runtime image plus `megatron-core`:

| Backend | Image |
| --- | --- |
| `nvidia-cuda133` | `harbor.baai.ac.cn/flagos-app/megatron_rl0.17.1-generic-13.3:2.2.0-0.2.3` |

Image names and tags are owned by build-infra. The table reflects the
`2.2.0-0.2.3` release; the current catalog is at
<https://flagos-ai.github.io/release-info/>.

Inside the image, install FlagTrain and its optional test or vendor dependencies
with `uv pip install`:

```bash
uv pip install -e .                  # runtime
uv pip install -e ".[test]"          # development and tests
uv pip install -e ".[test,nvidia]"   # NVIDIA dependencies and tests
```

> `uv` is a builtin command/feature in `build-infra` images

> **Important:** Use `uv pip install`, not `pip install`, when installing a
> vendor extra. DeepSpeed declares generic `torch` and `triton` dependencies,
> while FlagOS `build-infra` images already provide hardware-specific
> compatible builds. FlagTrain uses `[tool.uv].exclude-dependencies` to keep
> those image packages out of dependency resolution and still install
> DeepSpeed's other transitive dependencies. `pip` does not read this `uv`
> configuration and currently has no equivalent project setting for
> excluding selected transitive dependencies.

On the FlagOS network, set `UV_DEFAULT_INDEX` to the package mirror before
installing.

Vendor extras install the stack pinned in `backends.yaml` and are intended for
environments without a FlagOS image:

```bash
uv pip install -e ".[metax]"        # MetaX
uv pip install -e ".[test,nvidia]"  # NVIDIA + tests
```

The image already provides `torch`, `numpy`, `flagtree` and `megatron-core`. The `uv`
exclusion policy preserves the image-provided torch and Triton implementations
while resolving the remaining vendor-extra dependencies. Keep vendor extras in
sync with the versions documented in `backends.yaml`.

## Design Goals

- **Training computation in Triton**: Develop reusable operators for forward computation, backpropagation, and related compute modules in training.
- **Fusion and performance optimization**: Optimize operator fusion, memory access, and parallel computation for training workloads to reduce intermediate data transfers and execution overhead.
- **Multiple hardware backends**: Build on compatible Triton compilation and runtime environments to adapt functionality and tune performance across hardware backends.
- **Upstream collaboration**: Align with the interfaces, computational semantics, and usage requirements of target frameworks and components, and keep pace with upstream changes.
- **Verifiable implementations**: Provide correctness tests and performance benchmarks for operators and compute components, with clear applicability and validation scope.

## Component Scope

| Framework or Component | Areas of Focus |
|---|---|
| [Megatron](https://github.com/NVIDIA/Megatron-LM) | Key compute modules and related operators for large-model training |
| [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | Operators and fused computation for training optimization |
| [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) | Transformer compute modules and operators for mixed-precision and low-precision training |

These projects form the initial scope. Support will be documented by operator, data type, component version, and hardware backend in the [compatibility notes](docs/compatibility.md).

## Relationship with Related Projects

FlagTrain is developed in collaboration with operator libraries, compilers, and training frameworks in the FlagOS ecosystem, with distinct responsibilities:

- **[FlagGems](https://github.com/flagos-ai/FlagGems)**: Provides general-purpose operator implementations. FlagTrain prioritizes collaboration with FlagGems on reusable foundational operators to reduce duplicate implementations.
- **FlagTrain**: Focuses on operators, fused computation, and compute components for training, organizing their implementations, tests, and performance optimization.
- **Training frameworks and upstream components**: Provide use cases and references for interfaces and computational semantics. FlagTrain develops and validates Triton implementations for their key computation paths.

Specific invocation and integration methods will be described in the relevant module documentation and usage examples.

## Directory Structure

```text
FlagTrain/
├── src/flag_train/
│   ├── megatron/               # Megatron-related computation implementations
│   ├── deepspeed/              # DeepSpeed-related computation implementations
│   └── transformer_engine/     # Transformer Engine-related computation implementations
├── tests/                      # Correctness and interface compatibility tests by component
├── benchmark/                  # Performance benchmarks by component
├── examples/                   # Usage and integration examples by component
├── docs/                       # Installation, development, and compatibility notes
├── tools/                      # Development and maintenance tools
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE
└── pyproject.toml
```

The three component directories are intended to house their respective Triton computation implementations and interface wrappers. Within each component, files will be organized by operator or compute module; forward computation, backward computation, and invocation wrappers can live together. Shared functionality will be extracted when a clear need for reuse emerges.

## Development and Validation

Each operator or compute component implementation should include a corresponding reference implementation, test cases, and a method for evaluating performance.

### Correctness Testing

- Cover representative input shapes, data types, and edge cases.
- Validate results against the reference implementation and specify numerical error tolerances.
- Validate gradient computation for implementations involving backpropagation.
- Record the component versions, software environment, and hardware backend used in testing.

### Performance Testing

- Compare implementations under the same hardware, input, and precision conditions.
- Describe the reference implementation, warm-up procedure, and timing method.
- Report execution time, throughput, or device memory usage as appropriate for the computation.
- Distinguish individual operator performance from end-to-end training gains.

See [Getting Started](docs/getting_started.md) to install the project skeleton locally and the [development guide](docs/development.md) for development conventions. Operator usage examples and test commands will be provided with the first implementations.

## Contributing

Contributions to training operators, component adaptation, performance optimization, tests, and documentation are welcome.

- See the [contribution guide](CONTRIBUTING.md) for the contribution process.
- Report issues and suggest features through [GitHub Issues](https://github.com/flagos-ai/FlagTrain/issues).
- Report security issues privately by following the [security reporting instructions](SECURITY.md).

## License

This project is licensed under the [Apache License 2.0](LICENSE). Any included third-party code must retain its original license and copyright notices.
