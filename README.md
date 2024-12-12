# Geo Dataset

### Description

This package provide essential tools for cutting rasters and their labels into smaller tiles, useful for machine learning tasks. Also provides datasets compatible with pytorch.

### Installation

It is strongly advised to use this library on a Linux-based system.

\
First, install the following dependencies:

```bash
sudo apt update
sudo apt install -y build-essential ninja-build cmake python3-dev
```
\
Then, make sure you have [Rust](https://www.rust-lang.org/tools/install) installed on your system (it is needed for point cloud processing libraries).\
On linux, you can install it with:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

\
Finally, you can install the python dependencies with pip:

```bash
pip install git+ssh://git@github.com/hugobaudchon/geodataset.git
```

### Documentation

Documentation can be found here: https://hugobaudchon.github.io/geodataset/

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

