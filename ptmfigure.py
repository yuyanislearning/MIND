import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns
import argparse
import os
import ntpath

def gen_fig(orig, mutant, res_path, protid, mutation):

    # load json files
    orig_df = pd.DataFrame(columns=["PTM", "Orig_Prob"])
    mutant_df = pd.DataFrame(columns=["PTM", "Mutant_Prob"])
    with open(orig, "r") as f:
        orig = json.load(f)
    with open (mutant, "r") as f:
        mutant = json.load(f)

    # build dataframe from probability data
    orig_df = pd.DataFrame.from_dict(orig, orient="index", columns=["Orig_Prob"])
    mutant_df = pd.DataFrame.from_dict(mutant, orient="index", columns=["Mutant_Prob"])
    df = orig_df.merge(mutant_df, left_index=True, right_index=True)
    df["Effect"] = df["Mutant_Prob"].astype(float) - df["Orig_Prob"].astype(float)
    df.to_csv(os.path.join(res_path, f"{protid}_{mutation}.csv"))
    df = df.reset_index()
    df['PTM'] = df['index']
    df['index'] = df.index

    # plot data
    sns.set_theme(style='dark')
    sns.color_palette("muted")
    ax = sns.barplot(x='index', y='Effect', data=df, linewidth=0.1)
    sns.stripplot(x='index', y='Effect', data=df, s=2)
    ax.set(title="Impact of SNP on PTM Presence")
    ax.set(ylabel="probability effect")
    ax.set(xticklabels=[])
    ax.set(xlabel="PTM")
    ax.tick_params(bottom=False)
    
    plt.savefig(os.path.join(res_path, f"{protid}_{mutation}.png"), bbox_inches='tight', transparent=True)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_path", help="path to wild-type PTM predictions")
    parser.add_argument("--mutant_path", help="path to mutant PTM predictions")
    parser.add_argument("--res_path", help="path to save figure")
    args = parser.parse_args()
    fname = ntpath.basename(args.mutant_path)
    protid, mutation = fname.split('_')
    mutation, temp = mutation.split('.')
    
    gen_fig(args.orig_path, args.mutant_path, args.res_path, protid, mutation)
