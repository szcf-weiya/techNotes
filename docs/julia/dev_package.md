---
comments: true
---

# Workflow of Writing a Package

## Start with Empty

```julia
$ julia1.7
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.7.0 (2021-11-30)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

(@v1.7) pkg> generate MonoDecomp.jl
  Generating  project MonoDecomp:
    MonoDecomp.jl/Project.toml
    MonoDecomp.jl/src/MonoDecomp.jl

julia> 
```

## Starting with existing folder

1. create a github repo, say `xfun.jl`
2. open a Julia session, and then run `] generate xfun.jl`, which will generate a subfolder `xfun.jl`, obtain the resulting `Project.toml`, and wrap the source code by module as shown in the generated code `src/xfun.jl`. 
3. discard the generated `xfun.jl` subfolder.
4. two choices for adding dependencies for testing
    - use `extras` and `targets` 
    - use `test/Project.toml`
5. register package: follow the instruction at <https://github.com/JuliaRegistries/General#registering-a-package-in-general>
    - via the web interface: login with GitHub account, and then fill the form.
    - via <https://github.com/JuliaRegistries/Registrator.jl>
        - first click `install app`, then follow the instruction to grant the access to which repo.
        - comment `@JuliaRegistrator register` in the commit to be register

## Develop/Debug Locally

1. install the package via `dev ABSOLUTE_PATH` (usually in `v1.x` env). The absolute path (not relative path) can allow the installed package to be found starting from everywhere instead of the "relative" project.
2. make any changes on the source code, NO need to run `dev` again, the update will be automatically synced.
3. `test PACKAGE_NAME` (usually in `v1.x` env) would running the tests
4. activate the `docs` env, and run `include("make.jl")`, then open a http server, `python3 -m http.server 8080` under the `build/` folder

!!! warning "NOT `add LOCAL_PACKAGE`"
    Although the package can be installed, but the version is also controlled by git, that is, a uncommitted version cannot be updated.

    In that case, a new commit is also not automatically installed. `up LOCAL_PACKAGE` is necessary. Just imagine that adding package in this way is almost like to add packages remotely.

!!! example
    - <https://github.com/szcf-weiya/LaTeXTables.jl>
    - <https://github.com/szcf-weiya/DegreesOfFreedom.jl>

refer to

- [Generating a package with PkgTemplate for existing code](https://discourse.julialang.org/t/generating-a-package-with-pkgtemplate-for-existing-code/25163)
- [5. Creating Packages -- Pkg.jl](https://pkgdocs.julialang.org/v1/creating-packages/)
