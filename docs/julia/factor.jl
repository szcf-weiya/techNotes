function f(A)
    visited = Dict{Any, Int}()
    n = length(A)
    B = zeros(Int, n)
    count = 0
    for i = 1:n
        if get(visited, A[i], 0) > 0 # if A[i] has been visited, return the index, otherwise return 0
            B[i] = visited[A[i]]
        else
            count += 1
            B[i] = count
            visited[A[i]] = B[i]
        end
    end
    return B
end