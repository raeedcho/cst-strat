import mat73
import pandas as pd
import pyaldata
import matplotlib.pyplot as plt
import matplotlib as mpl

# this is unfinished...
def plot_sensorimotor(trial, ax=None, color_by_state=False):
    if ax is None:
        ax = plt.gca()

    plt.plot([-60,60],[60,-60],'--k')
    plt.plot([0,0],[-60,60],'-k')
    plt.plot([-60,60],[0,0],'-k')
    if color_by_state:
        plt.scatter(
            trial['cursor_pos'][:,0],
            trial['hand_pos'][:,0],
            c=hmm_z,
            cmap=cmap,
            s=10,
            norm=mpl.colors.Normalize(vmin=0,vmax=len(color_names)-1)
            )
    else:
        plt.scatter(trial['cursor_pos'][:,0],trial['hand_pos'][:,0],c='k',s=10)
        plt.xticks([])
        plt.yticks([])
        plt.xlim(-scale,scale)
        plt.ylim(-scale,scale)
        plt.xlabel('Cursor position')
        plt.ylabel('Hand position')
        sns.despine(left=True,bottom=True)
