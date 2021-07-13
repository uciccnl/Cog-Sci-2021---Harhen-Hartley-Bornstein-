using CSV
using DataFrames
using Random, Distributions
using JLD

include("./model_struct.jl")

function overharvest(opt_prt_all,opt_prt)
    append!(opt_prt_all,opt_prt)
    return opt_prt_all
end

function underharvest(v_stay, v_leave, opt_prt_all, opt_prt, reward, total_reward, total_time, galaxy, exp)
    temp_tot_r = total_reward
    temp_tot_t = total_time

    while v_stay > v_leave
        opt_prt+= 1
        decay = sample_decay(galaxy)
        reward = reward*decay
        temp_tot_r += reward
        temp_tot_t += exp.harvest_time + exp.iti

        pred_decay = get_decay(galaxy)
        v_stay = reward*pred_decay
        v_leave = (temp_tot_r/temp_tot_t)*(exp.harvest_time + exp.iti)
    end
    append!(opt_prt_all,opt_prt)
    return opt_prt_all
end

function sample_decay(galaxy)
    if galaxy == 0
        a = 13
        b = 51
    elseif galaxy == 1
        a = 50
        b = 50
    elseif galaxy == 2
        a = 50
        b = 12
    end
    d = Beta(a,b)
    return rand(d)
end

function get_decay(galaxy)
    #println("galaxy")
    #println(galaxy)
    if galaxy == 0
        return 0.2
    elseif galaxy == 1
        return 0.5
    elseif galaxy == 2
        return 0.8
    end
end

function optimal_policy(behavior)
    exp = Experiment(5,360,20,2,10,5.5,1.5)
    total_reward = 0
    total_time = exp.alien_time + exp.iti
    n_trials = length(behavior.prt)

    opt_prt_all = [] # to fill
    for i in 1:n_trials
        planet_rewards = behavior.reward_list[i]
        galaxy = behavior.galaxy[i]
        true_prt = behavior.prt[i]
        opt_prt = 0
        not_appended = true

        for j in 1:true_prt
            opt_prt += 1
            # get reward
            reward = planet_rewards[j]
            total_reward += reward
            total_time += exp.harvest_time + exp.iti
            # should i stay or should i go?
            pred_decay = get_decay(galaxy)
            v_stay = reward*pred_decay
            v_leave = (total_reward/total_time)*(exp.harvest_time + exp.iti)

            # overharvesting
            if (v_stay <= v_leave) & (not_appended)
                append!(opt_prt_all,opt_prt)
                not_appended= false
            end
            # underharvesting
            if (v_stay > v_leave) & (j == true_prt)
                opt_prt_all = underharvest(v_stay, v_leave, opt_prt_all, opt_prt, reward, total_reward,total_time, galaxy, exp)
                not_appended= false
            end
        end
        # leave planet
        total_time += exp.travel_time + exp.alien_time
    end
    return opt_prt_all
end
