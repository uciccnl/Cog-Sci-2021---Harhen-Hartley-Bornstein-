using StatsBase
using JLD
using DataFrames
using CSV

include("./model.jl")
include("./fit_model.jl")

function get_block(true_planet)
    block =[]
    for p in true_planet
        if p < 20
            append!(block,1)
        elseif p < 40
            append!(block,2)
        elseif p < 60
            append!(block,3)
        elseif p < 80
            append!(block,4)
        elseif p < 100
            append!(block,5)
        end
    end
    return block
end


function sub_params(sub_num,block,alpha)
    if alpha
        d = load(string("fit_params/alpha_free/sub",string(sub_num),"_testBlock",string(block),".jld"))
        params = d["res"]
    else
        d = load(string("fit_params/alpha_0/sub",string(sub_num),"_testBlock",string(block),".jld"))
        params = d["res"]
        params = [0,params[1]]
    end

    num_particles = 5
    sub_data = get_sub_data(sub_num)
    b,df = short_ref_point(sub_data,params,num_particles);
    opt_prt = optimal_policy(b);
    diff = b.prt - opt_prt;

    df = DataFrame(Dict("true_planet"=> b.true_planet,"galaxy"=> b.galaxy,"block"=> get_block(b.true_planet),"prt"=>b.prt,
        "opt_prt"=>opt_prt, "diff" => diff))

    gdf = groupby(df, :galaxy)
    prt_avg = combine(gdf, :diff => mean)

    insertcols!(prt_avg,
    1,
    :sub_num => ones(size(prt_avg)[1])*sub_num,
    makeunique=true)

    insertcols!(prt_avg,
    2,
    :block => ones(size(prt_avg)[1])*block,
    makeunique=true)
    return prt_avg
end

function all_subs(subs,block,alpha)
    df_prt = DataFrame()
    for sub in subs
        println(sub)
        prt = sub_params(sub,block,alpha)
        df_prt = vcat(df_prt,prt)
    end
    return df_prt
end

function main()
    alpha = parse(Int64,ARGS[1])
    if alpha == 1
        fname = "results/fit_results_alpha_free.csv"
        alpha = true
    else
        fname = "results/fit_results_alpha_0.csv"
        alpha = false
    end

    subs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
    prt = DataFrame()
    for block in 1:5
        block_prt = all_subs(subs,block,alpha)
        prt = vcat(prt,block_prt)
    end
    CSV.write(fname,prt)

end

main()
