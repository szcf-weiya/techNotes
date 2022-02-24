# Workflow of Writing a Package

## Starting with existing folder

1. create a github repo, say `xfun.jl`
2. open a Julia session, and then run `] generate xfun.jl`, which will generate a subfolder `xfun.jl`, obtain the resulting `Project.toml`, and wrap the source code by module as shown in the generated code `src/xfun.jl`. 
3. discard the generated `xfun.jl` subfolder.
4. two choices for adding dependencies for testing
    - use `extras` and `targets` 
    - use `test/Project.toml`
5. register package: follow the instruction at <https://github.com/JuliaRegistries/General#registering-a-package-in-general>
    - use <https://github.com/JuliaRegistries/Registrator.jl>
    - first click `install app`
    - via the web interface: login with GitHub account, and then fill the form.

- local package can be added via the path, but note that the version is also controlled by git, that is, a uncommitted version cannot be updated.

refer to

- [Generating a package with PkgTemplate for existing code](https://discourse.julialang.org/t/generating-a-package-with-pkgtemplate-for-existing-code/25163)
- [5. Creating Packages -- Pkg.jl](https://pkgdocs.julialang.org/v1/creating-packages/)
- Example: [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl)

## Update package

1. update code and add related test cases
2. the package is added via local path instead of github repo
3. run `git add` and `git commit` to let the changes take effects
4. update xfun in julia, `up xfun`
4. open a new julia session, and then run `using xfun`, which will load the new commits
5. run the tests from command line
6. finally, push the new commits