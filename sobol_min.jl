using Sobol
include("./model.jl")
include("./box.jl")



function minimize(func, samp_points)
    all_loss = []
    params_tried = []

    for i in 1:length(samp_points)
        params = samp_points[i]
        loss = func(params)
        append!(all_loss,loss)
        append!(params_tried,[params])
    end
    min_ind = argmin(all_loss)
    return params_tried[min_ind]
end

function sobol_min(func)
    box = Box(
        a = (0, 50, :log),
        b = (1, 28)
    )
    N = 1000
    xs = Iterators.take(SobolSeq(n_free(box)), N) |> collect
    result = minimize(func, xs)
    return result
end
