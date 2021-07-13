using CSV
include("./model.jl")
include("./fit_model.jl")

function main()
    sub_num = parse(Int64,ARGS[1])
    sub_data = get_sub_data(sub_num)

    alpha = parse(Float64,ARGS[2])
    env_init = parse(Float64,ARGS[3])
    params = [alpha, env_init]

    num_particles = 5

    behav, track_inference = short_ref_point(sub_data,params,num_particles)
    prt_avg = package_results(behav)

    CSV.write(string("results/sub_num_",string(sub_num),"_alpha_",string(alpha),"_envInit_",string(env_init),"_prt.csv"),prt_avg)
    CSV.write(string("results/sub_num_",string(sub_num),"_alpha_",string(alpha),"_envInit_",string(env_init),"_trackInference.csv"),track_inference)
end

main()
