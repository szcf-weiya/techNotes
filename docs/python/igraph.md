# IGRAPH

tutorial: https://igraph.org/python/doc/tutorial/tutorial.html

https://stackoverflow.com/questions/25303620/does-igraphs-gomory-hu-tree-calculate-the-minimum-cut-tree

最小割：原图的每条边有一个割断它的代价，你需要用最小的代价使得这两个点不连通

## Installation

### first attempt

```bash
pip install python-igraph
```

but cannot plot, and it throws,

```python
AttributeError: Plotting not available; please install pycairo or cairocffi
```

then 

```bash
pip install pycairo
```

but it failed with the following error

```bash
  Complete output (15 lines):
  running bdist_wheel
  running build
  running build_py
  creating build
  creating build/lib.linux-x86_64-3.8
  creating build/lib.linux-x86_64-3.8/cairo
  copying cairo/__init__.py -> build/lib.linux-x86_64-3.8/cairo
  copying cairo/__init__.pyi -> build/lib.linux-x86_64-3.8/cairo
  copying cairo/py.typed -> build/lib.linux-x86_64-3.8/cairo
  running build_ext
  Package cairo was not found in the pkg-config search path.
  Perhaps you should add the directory containing `cairo.pc'
  to the PKG_CONFIG_PATH environment variable
  No package 'cairo' found
  Command '['pkg-config', '--print-errors', '--exists', 'cairo >= 1.15.10']' returned non-zero exit status 1.
  ----------------------------------------
  ERROR: Failed building wheel for pycairo
Failed to build pycairo
ERROR: Could not build wheels for pycairo which use PEP 517 and cannot be installed directly
```

### second attempt

```bash
conda install -c conda-forge python-igraph
conda install -c conda-forge pycairo
```

note that the second command would also install the missing `cairo`,

```bash
The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    cairo-1.16.0               |       hf32fb01_1         1.0 MB
    fontconfig-2.13.1          |       h6c09931_0         250 KB
    glib-2.68.3                |       h9c3ff4c_0         449 KB  conda-forge
    glib-tools-2.68.3          |       h9c3ff4c_0          86 KB  conda-forge
    libglib-2.68.3             |       h3e27bee_0         3.1 MB  conda-forge
    pycairo-1.20.1             |   py38hf61ee4a_0          77 KB  conda-forge
    ------------------------------------------------------------
                                           Total:         5.0 MB
```

Thus, plotting can be done.

## Growing Minimum spanning tree

The generic method manages a set of edges $A$, maintaining the following loop invariant

> Prior to each iteration, $A$ is a subset of some minimum spanning tree.

At each step, determine an edge $(u, v)$ that we can add to $A$ without violating this invariant, in the sense that $A\cup \\{(u, v)\\}$ is also a subset of a minimum spanning tree. Such as edge is called **safe edge** for $A$.

A 
