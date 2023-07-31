function f()
    return [1, 2, 3]
end

const c = f()

function g()
    return c[1]
end

function h(i)
    return c[i]
end