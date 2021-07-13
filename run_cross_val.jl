using StatsBase
using JLD
using DataFrames
using CSV

include("./model.jl")
include("./fit_model.jl")


function get_sub_ref_point(sub_num)
    all_data=DataFrame(CSV.File("behavioral_data/ref_point_by_block.csv",delim=','))
    sub_data = all_data[in(sub_num).(all_data.sub_num),:]

    return sub_data
end

function cross_val_func(sub_num,alpha)
    sse = 0
    for block = 1:5
        if alpha
            d = load(string("fit_params/alpha_free/sub",string(sub_num),"_testBlock",string(block),".jld"))
            params = d["res"]
            params = [params[1],params[2]]
        else
        d = load(string("fit_params/alpha_0/sub",string(sub_num),"_testBlock",string(block),".jld"))
            params = d["res"]
            params = [0,params[1]]
        end
        sub_data = get_sub_data(sub_num)
        num_particles = 5
        all_pred = []
        for i = 1:30
            b =short_ref_point(sub_data,params,num_particles)

            opt_prt = optimal_policy(b);
            diff = b.prt - opt_prt;

            df = DataFrame(Dict("true_planet"=> b.true_planet,"block"=> get_block(b.true_planet),"galaxy"=> b.galaxy,"prt"=>b.prt,
            "opt_prt"=>opt_prt, "diff" => diff))

            gdf = groupby(df, [:block,:galaxy])
            prt_avg = combine(gdf, :diff => mean)
            pred = prt_avg[in(block).(prt_avg.block),:].diff_mean[1]
            append!(all_pred,pred)
        end

        pred = mean(all_pred)

        ref_point = get_sub_ref_point(sub_num)
        target = ref_point[in(block).(ref_point.block),:].prt_rel_om[1]
        square_error = (pred-target)^2
        sse += square_error
    end
    return sse
end

function all_cross_val(subs)
    sse_alpha_free = []
    sse_alpha_0 = []

    for sub in subs
        append!(sse_alpha_free, cross_val_func(sub,true))
        append!(sse_alpha_0, cross_val_func(sub,false))
    end
    df = DataFrame(Dict("sub"=>subs,"alpha_free"=> sse_alpha_free,"alpha_0"=>sse_alpha_0))
    return df
end

function main()
    subs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
    df = all_cross_val(subs)
    CSV.write("results/cross_val_results.csv",df)
end

main()
