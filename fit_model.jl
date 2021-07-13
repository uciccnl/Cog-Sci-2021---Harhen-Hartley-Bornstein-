using JLD


include("./model.jl")
include("./opt.jl")
include("./sobol_min.jl")

function get_sub_data(sub_num)
    all_data=DataFrame(CSV.File("behavioral_data/all_data.csv",delim=','))
    sub_data = all_data[in(sub_num).(all_data.sub_num),:]
end

function get_sub_ref_point(sub_num)
    # gets the key summary statistics of the participant's behavior into the
    # correct format for comparison to model generated behavior summary statistics
    all_data=DataFrame(CSV.File("behavioral_data/ref_point_by_block.csv",delim=','))
    sub_data = all_data[in(sub_num).(all_data.sub_num),:]

    galaxies = [0,1,2]
    all_prt_rel_opt = []
    all_blocks = []
    all_galaxies = []
    for block in 1:5
        block_data = sub_data[in(block).(sub_data.block),:]
        for galaxy in galaxies
            galaxy_data = block_data[in(galaxy).(block_data.galaxy),:]
            if nrow(galaxy_data) > 0
                append!(all_prt_rel_opt,galaxy_data.prt_rel_om)
                append!(all_blocks,block)
                append!(all_galaxies,galaxy)
            end
        end
    end
    return DataFrame(Dict("block"=>all_blocks,"galaxy"=>all_galaxies,"prt_rel_opt"=>all_prt_rel_opt))
end

function get_block(true_planet)
    block =[]
    # max number of planets a participant could visit in a block is 20 based on the time limits, harvest/dig time, and travel time
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


function package_results(behavior)
    opt_prt = optimal_policy(behavior) # Marginal Value Theorem optimal
    diff = behavior.prt - opt_prt # difference between behavior (subject or model) and optimal policy according to MVT

    # put in dataframe and save
    # participants will visit a different number of planets depending on their harvesting behavior (number of stays on each planet), so true planet is
    # way to track between participants their behavior on the same "true" planet
    df = DataFrame(Dict("true_planet"=> behavior.true_planet,"block"=> get_block(behavior.true_planet),"galaxy"=> behavior.galaxy,"prt"=>behavior.prt,
        "opt_prt"=>opt_prt, "diff" => diff))

    # get the average PRT (planet residence time) relative to MVT optimal
    gdf = groupby(df, [:block,:galaxy])
    prt_avg = combine(gdf, :diff => mean)
    return prt_avg
end

function return_loss(behavior,ref_points,test_block)
    prt_avg = package_results(behavior) # get model generated data in same format as the subject data
    galaxies = [0,1,2] # 0 = poor, 1 = neutral, 2 = rich
    blocks = [1,2,3,4,5]
    filter!(e->e≠test_block,blocks) # remove the block that we will use as a test later, we do not want to fit on this block
    sse = 0 # summed square error
    n_datapoints = 0
    for block in blocks
        for galaxy in galaxies
            pred = prt_avg[(prt_avg[!,"block"].==block),:] # model prediction, select on block
            target = ref_points[(ref_points[!,"block"].==block),:] # subject data, select on block

            pred = pred[(pred[!,"galaxy"].==galaxy),:].diff_mean # further filter on galaxy and then get the PRT (planet residence time) relative to MVT, this is what diff_mean references
            target = target[(target[!,"galaxy"].==galaxy),:].prt_rel_opt# further filter on galaxy and then get the PRT (planet residence time) relative to MVT, this is what prt_rel_opt references

            if (length(pred) > 0) & (length(target) > 0) # need to check because not every planet type will be encountered in every block
                error = (pred[1] - target[1])^2 # squared error
                sse += error
                n_datapoints += 1
            end
        end
    end
    return sse/n_datapoints
end

function loss_function(params,n_sims=30,num_particles=5)
    sub_num = parse(Int64,ARGS[1])
    test_block = parse(Int64,ARGS[2])

    sub_data = get_sub_data(sub_num) # get the rewards experienced on each planet for this subject
    sub_ref_points = get_sub_ref_point(sub_num) # get the subject's prt for each planet type in each block
    all_sse = 0
    for i in 1:n_sims
        b = short_ref_point(sub_data,params,num_particles) # model generated behavior
        all_sse += return_loss(b,sub_ref_points,test_block) # compare model behavior with subject behavior
    end
    return all_sse/n_sims # mse across all sims
end


function opt_params()
    println("start opt")
    res = sobol_min(loss_function)
    return res
end

function main()
    sub_num = parse(Int64,ARGS[1])
    test_block = parse(Int64,ARGS[2])
    result = opt_params()
    println(result)
    save(string("fit_params/sub",string(sub_num),"_testBlock",string(test_block),".jld"), "res", result)
end

#main()
