mutable struct MemBuffer{X,Y}
    x::Vector{X}
    y::Vector{Y}
    capacity::Int
    i::Int
end

MemBuffer{X,Y}(cap::Int) where {X,Y} = MemBuffer(X[],Y[], cap, 0)

function Base.getindex(mem::MemBuffer, i)
    @boundscheck checkbounds(mem.x, i)
    @inbounds return (mem.x[i], mem.y[i])
end

Base.length(mem::MemBuffer) = length(mem.x)

function Base.push!(mem::MemBuffer{X,Y}, x::X, y::Y) where {X,Y}
    i = (mem.i += 1)
    k = mem.capacity
    if i ≤ k
        push!(mem.x,x)
        push!(mem.y,y)
    else
        j = rand(1:i)
        if j ≤ k
            mem.x[j] = x
            mem.y[j] = y
        end
    end
end

function Base.empty!(mem::MemBuffer)
    empty!(mem.x)
    empty!(mem.y)
    mem.i = 0
end
