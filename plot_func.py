import matplotlib.pyplot as plt
import seaborn as sns

def get_marker_size(variance):
    if variance < 0.003:
        return 5
    elif variance < 0.005:
        return 6
    elif variance < 0.007:
        return 7
    elif variance < 0.01:
        return 9
    elif variance < 0.03:
        return 13
    elif variance < 0.05:
        return 17
    elif variance < 0.1:
        return 20
    elif variance < 0.2:
        return 25

galaxy_color = {0:[1,0.1,0],1:[0,0.3,0.9],2:[0,0.6,0.2]}


def plot_infer_decay(sims):
    sims['planet'] = range(len(sims))
    for n_trial in range(len(sims)):
        cluster_mean = sims.query("planet=="+str(n_trial)).cluster_mean.reset_index(drop=True)[0]
        cluster_var = sims.query("planet=="+str(n_trial)).cluster_var.reset_index(drop=True)[0]
        true_decay = sims.query("planet=="+str(n_trial)).true_decay.reset_index(drop=True)[0]
        plt.plot(n_trial,cluster_mean,'o',markersize=get_marker_size(cluster_var),color=[0,0,0,0.2])
        true_galaxy = sims.query("planet=="+str(n_trial)).true_galaxy.reset_index(drop=True)[0]
        plt.plot(n_trial,true_decay,'x',markersize=15,color=galaxy_color[true_galaxy])
    plt.ylim([0,1])
    plt.xlabel("planet #")
    plt.ylabel('decay rate')
    return

def plot_ref_point(sims):
    sims['planet'] = range(len(sims))
    for n_trial in range(len(sims)):
        mean = sims.query("planet=="+str(n_trial)).short_alpha.reset_index(drop=True)[0]/sims.query("planet=="+str(n_trial)).short_beta.reset_index(drop=True)[0]
        var = sims.query("planet=="+str(n_trial)).short_alpha.reset_index(drop=True)[0]/(sims.query("planet=="+str(n_trial)).short_beta.reset_index(drop=True)[0]**2)
        plt.plot(n_trial,mean,'o',markersize=get_marker_size(var),color=[0,0,0,0.2])
        mvt = sims.query("planet=="+str(n_trial)).long_alpha.reset_index(drop=True)[0]/sims.query("planet=="+str(n_trial)).long_beta.reset_index(drop=True)[0]
        plt.plot(n_trial,mvt,'x',markersize=15,color='k')
    plt.xlabel("planet #")
    plt.ylim([0,10])
    plt.ylabel('reward rate')
    plt.title('reference point')
    return
