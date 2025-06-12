# GeoDataset

### Description

This package provide essential tools for cutting rasters and their labels into smaller tiles, useful for machine learning tasks. Also provides datasets compatible with pytorch.

### Documentation

Documentation can be found here: https://hugobaudchon.github.io/geodataset/

### Installation

It is strongly advised to use this library on a Linux-based system.

First, make sure you have the required dependencies installed\:

```bash
sudo apt update
sudo apt install -y build-essential ninja-build cmake python3-dev
```

#### Basic installation
Then, install the library with pip:

```bash
pip install git+https://github.com/hugobaudchon/geodataset.git
```

or for a specific version like for example v0.4.1:
    
```bash
pip install git+https://github.com/hugobaudchon/geodataset.git@v0.4.1
```

#### Additional Point Cloud support

⚠️ *Still in development.*

To install geodataset with point cloud support, make sure you have [Rust](https://www.rust-lang.org/tools/install) installed on your system, which is required for the lazpy library.\
On linux, you can install it with:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```


Finally, you can install the library with point cloud support by adding the `[pc]` extra:

```bash
pip install git+https://github.com/hugobaudchon/geodataset.git[pc]
```

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

