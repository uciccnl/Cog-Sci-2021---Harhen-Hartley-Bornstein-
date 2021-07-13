using CSV
using DataFrames
using Random, Distributions
using JLD
using StatsBase

include("./particle_filter.jl")
include("./model_struct.jl")
include("./opt.jl")

#################### MODEL HELPER FUNCTIONS ########################################
function get_sub_data(sub_num)
    all_data=DataFrame(CSV.File("behavioral_data/all_data.csv",delim=','))
    sub_data = all_data[in(sub_num).(all_data.sub_num),:]
end

function calc_posterior(decay_list, N, alpha, particle_filter)
    num_particles = length(particle_filter)
    posterior = [[] for i = 1:num_particles]
    for p in 1:num_particles
        # 1. get log prior
        log_prior = log_prior_particle(particle_filter[p],N,alpha)
        # 2. get log likelihood
        log_like = log_likelihood_particle(particle_filter[p],decay_list)
        # 3. combine to get posterior
        post = exp.(log_prior .+ log_like)

        post_norm = post/sum(post)
        posterior[p] = post_norm
    end
    return posterior
end


function prob_cluster(curr_reward, decay_list, N, alpha, particle_filter)
    if (N == 0) | (length(decay_list) == 0)
        return prior(particle_filter, N , alpha)
    else
        return calc_posterior(decay_list, N, alpha, particle_filter)
    end
end


function get_last_galaxy_identity(cluster_assign,current_cluster)
    ind_differ = [i for i in 1:length(cluster_assign) if cluster_assign[i] != current_cluster]
    if length(cluster_assign) == 0
        last_cluster = 1
    elseif length(ind_differ) > 0
        last_cluster_ind=maximum(ind_differ)
        last_cluster = cluster_assign[last_cluster_ind]
    else
        last_cluster = current_cluster
    end
    return last_cluster
end

function sample_v_stay(particle_filter,weights, prob_k, n_samples, data)
    values = zeros(n_samples)
    for s in 1:n_samples
        # 1. sample a particle
        cum_weight = cumsum(weights)
        sampled_particle = argmax(rand() .< cum_weight) # try another way to make sure it works

        # 2. sample a cluster
        cum_particle_post = cumsum(prob_k[sampled_particle])
        cluster = argmax(rand() .< cum_particle_post) # try another way to make sure it works

        # 3. sample from the distribution associated with this particle's cluster that was sampled
        if cluster > length(particle_filter[sampled_particle].cluster_mean)
            # create a new partilce just for the purpose of getting mu and sigma
            mu = particle_filter[sampled_particle].hyper_mu
            sigma = particle_filter[sampled_particle].hyper_var^(1/2)
        else
            # might want to conform expectation to  the data
            mu = particle_filter[sampled_particle].cluster_mean[cluster]
            sigma = particle_filter[sampled_particle].cluster_variance[cluster]^(1/2)
        end

        d = Normal(mu,sigma)
        sampled_value = rand(d)
        values[s] = sampled_value
    end
    return mean(values)
end


function sample_v_leave_ref_point(particle_filter,particle_weights, prob_k, n_samples)
    values = zeros(n_samples)
    for s in 1:n_samples
        # 1. sample a particle
        cum_weight = cumsum(particle_weights)
        sampled_particle = argmax(rand() .< cum_weight) # try another way to make sure it works

        # 2. sample a cluster
        cum_particle_post = cumsum(prob_k[sampled_particle])
        cluster = argmax(rand() .< cum_particle_post) # try another way to make sure it works

        # 3. get transition probabilities
        cluster_assign = particle_filter[sampled_particle].cluster_assign
        last_galaxy = get_last_galaxy_identity(cluster_assign,cluster)
        rr = particle_filter[sampled_particle].planet_reward[last_galaxy]/particle_filter[sampled_particle].planet_time[last_galaxy]

        # 5. take a weighted average
        values[s] = rr
    end
    return mean(values)
end

function get_galaxy(block_num, planet_num)
    exp_struc = DataFrame(CSV.File("exp_struc.csv",delim=','))

    tmp_1 = exp_struc[in(block_num).(exp_struc.block),:]

    tmp_2 = tmp_1[in(planet_num).(tmp_1.planet),:]

    galaxy = tmp_2.galaxy[1]

    return galaxy
end


function get_planet(block_num, block_tracker,data)
    last_planet = maximum(data.planet_in_block)

    if block_tracker.planet > last_planet
        curr_planet = DataFrame(sub_num = Int64[], block = Int64[],true_planet = Int64[], planet = Int64[],
            galaxy = Int64[], stay_num = Int64[], reward = Int64[])
        galaxy = get_galaxy(block_num,block_tracker.planet)
        p_tracker = Planet_Tracker(true,galaxy,0,0,(block_num-1)*20 + block_tracker.planet,[],[])

    else
        curr_planet = data[in(block_tracker.planet).(data.planet_in_block),:]
        galaxy = curr_planet.galaxy[1]
        p_tracker = Planet_Tracker(true,galaxy,0,0,(block_num-1)*20 + block_tracker.planet,[],[])
    end
    return curr_planet, p_tracker
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

function get_rho0()
    d = Normal(100,5)
    return rand(d)
end


function get_reward(planet_tracker, total_reward, data)
    # check if empty dataframe
    if (size(data.stay_num)[1] == 0)
        if planet_tracker.prt == 0
            last_reward = get_rho0()
            reward = round(last_reward)
        else
            last_reward = planet_tracker.curr_reward
            decay = sample_decay(planet_tracker.galaxy)
            reward = round(last_reward*decay)
        end
    else
        max_prt =  maximum(data.stay_num)
        if planet_tracker.prt > max_prt
             decay = sample_decay(planet_tracker.galaxy)
             reward = round(planet_tracker.curr_reward*decay)
         else
             # get the experienced reward
             reward = data[in(planet_tracker.prt).(data.stay_num),:].reward[1]
             decay = reward/planet_tracker.curr_reward
         end

    end

    if planet_tracker.prt > 0
        append!(planet_tracker.decay_list,decay)
    end

    planet_tracker.curr_reward = reward
    append!(planet_tracker.reward_list,reward)
    total_reward += reward
    return planet_tracker, total_reward
end

function make_choice(v_stay, v_leave, planet_tracker)
    if v_stay > v_leave
        planet_tracker.on_planet = true
    else
        planet_tracker.on_planet = false
    end
    planet_tracker.prt += 1 # need to add one to counteract the first dig even if leave
    return planet_tracker
end


function update_behavior(behav, planet_tracker)
    append!(behav.true_planet,planet_tracker.true_planet)
    append!(behav.galaxy,planet_tracker.galaxy)
    append!(behav.prt,planet_tracker.prt)
    append!(behav.reward_list,[planet_tracker.reward_list])
    return behav
end
#############################   MODEL  ######################################

function short_ref_point(data,params,num_particles,track_inference=true)
    # intialize the experiment
    exp = Experiment(5,360,20,2,10,5.5,1.5)
    behav = Behavior([],[],[],[])

    # initialize the particle filter
    alpha,env_init = params # free parameters that need to be fit
    hyper_mu = 0.5
    hyper_var = 0.25
    hyper_tau = 1
    n_samples = 1000
    particles, weights = init_particle_filter(num_particles,hyper_mu, hyper_var, hyper_tau, alpha,env_init)

    # initialize for tracking progress through the experiment
    total_reward = 0
    total_time = exp.alien_time + exp.iti
    N = 0 # all planets experienced, ***check if changes if I change to number of decay rates ***

    if track_inference
        track_cluster_assign, track_cluster_mu, track_cluster_var, track_short_alpha, track_short_beta, track_long_alpha, track_long_beta, true_decay, true_galaxy = [[] for _ = 1:9]
        all_decay = [[],[],[]]
    end
    for b in 1:exp.n_blocks
        curr_block = data[in(b).(data.block),:]
        b_tracker = Block_Tracker(exp.alien_time,0)
        while (b_tracker.block_time < exp.block_max_time) & (b_tracker.planet < exp.block_n_planet)
            curr_planet, p_tracker = get_planet(b, b_tracker,curr_block)
            while (b_tracker.block_time < exp.block_max_time) & p_tracker.on_planet

                # dig up reward and add to list
                p_tracker, total_reward = get_reward(p_tracker, total_reward, curr_planet)

                # add time that it took to dig it up
                b_tracker.block_time += exp.harvest_time + exp.iti
                total_time += exp.harvest_time + exp.iti

                # probability of cluster
                global prob_k = prob_cluster(p_tracker.curr_reward, p_tracker.decay_list, N, alpha, particles)

                # estimate the value of leaving and the value of staying
                v_stay = sample_v_stay(particles,weights, prob_k, n_samples,p_tracker.decay_list)*p_tracker.curr_reward
                v_leave = sample_v_leave_ref_point(particles,weights, prob_k, n_samples)*(exp.harvest_time + exp.iti)

                # take  the max of v_stay and v_leave
                p_tracker = make_choice(v_stay, v_leave,p_tracker)
            end
            planet_reward = sum(p_tracker.reward_list)
            planet_time = p_tracker.prt*(exp.harvest_time + exp.iti) +
                        exp.travel_time + exp.alien_time
            if p_tracker.prt > 1
                particles, weights = resample_and_update_particles(particles,
                    weights,prob_k,p_tracker.decay_list,planet_reward, planet_time)

                if track_inference
                    infer_cluster = particles[1].cluster_assign[end]
                    last_cluster = get_last_galaxy_identity(particles[1].cluster_assign,infer_cluster)

                    all_decay[p_tracker.galaxy+1] = vcat(all_decay[p_tracker.galaxy+1],p_tracker.decay_list)

                    append!(track_cluster_assign,infer_cluster)
                    append!(track_cluster_mu,particles[1].cluster_mean[infer_cluster])
                    append!(track_cluster_var,particles[1].cluster_variance[infer_cluster])

                    append!(track_short_alpha,particles[1].planet_reward[last_cluster])
                    append!(track_short_beta,particles[1].planet_time[last_cluster])

                    append!(track_long_alpha,total_reward)
                    append!(track_long_beta,total_time)
                    append!(true_decay,mean(all_decay[p_tracker.galaxy+1]))
                    append!(true_galaxy,p_tracker.galaxy)
                end
            end
            b_tracker.block_time += exp.travel_time + exp.alien_time
            b_tracker.planet += 1
            total_time += exp.travel_time + exp.alien_time
            behav = update_behavior(behav, p_tracker)
            N += 1
        end
    end
    if track_inference
        df = DataFrame(Dict("cluster_assign"=>track_cluster_assign,"true_galaxy"=>true_galaxy,"cluster_mean"=> track_cluster_mu,"cluster_var"=> track_cluster_var,"short_alpha"=>track_short_alpha,
        "short_beta"=>track_short_beta, "long_alpha" => track_long_alpha,"long_beta" => track_long_beta,"true_decay"=>true_decay))
        return behav,df
    else
        return behav
    end
end
