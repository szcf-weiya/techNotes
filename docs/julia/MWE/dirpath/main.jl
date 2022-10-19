using RCall
using PyCall


println("R: ", rcopy(R"$(@__DIR__)"))
println("Python", py"$(@__DIR__)")
println("Julia:", @__DIR__)

println()
println("Assign @__DIR__ to a variable:")
println()

current_folder = @__DIR__

println("R: ", rcopy(R"$(current_folder)"))
println("Python", py"$(current_folder)")
println("Julia:", current_folder)
