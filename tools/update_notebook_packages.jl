if !isdir("pluto-slider-server-environment") || length(ARGS) != 2
    error("""
    Run me from the root of the repository directory, using:

    julia tools/update_notebook_packages.jl <level> <run_notebooks>
    
    Where <level> is one of: PATCH, MINOR, MAJOR
    And <run_notebooks> is true or false to run all notebooks with Pluto at the end
    """)
end

if !(v"1.12.0-aaa" < VERSION < v"1.13.0")
    error("Our notebook package environments need to be updated with Julia 1.12. Go to julialang.org/downloads to install it.")
end

import Pkg
Pkg.activate("./pluto-slider-server-environment")
Pkg.instantiate()

import Pluto

flatmap(args...) = vcat(map(args...)...)


getfrom(dir) = flatmap(walkdir(dir)) do (root, _dirs, files)
    joinpath.((root,), files)
end

all_files_recursive = [getfrom("lectures")..., getfrom("probprog")..., getfrom("minis")...]

all_notebooks = filter!(Pluto.is_pluto_notebook, all_files_recursive)

level = getfield(Pkg, Symbol("UPLEVEL_$(ARGS[1])"))
run_notebooks = parse(Bool, ARGS[2])

for n in all_notebooks
    @info "Updating" n
    ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0
    Pluto.update_notebook_environment(n; backup=false, level)
end

@info "All notebooks done!"

if run_notebooks
    @info "Running all notebooks with Pluto..."
    Pluto.run(notebook=all_notebooks)
end
