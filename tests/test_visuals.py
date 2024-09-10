# @author NoCreativeIdeaForGoodUserName
# encoding: utf-8
# module deepof

"""

Testing module for deepof.visuals

"""

import os
import pickle
import seaborn as sns
from shutil import rmtree
import numpy as np
import pandas as pd
import deepof.data
import deepof.utils
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import use as use_backend
from PIL import Image



import deepof.data


#      TESTING PLOTS WITH SAVED EXAMPLES      #


###############################################
#         Create projects for testing         #
###############################################


#def create_ex_conds(cond_dict):
#    """creates exp_conditions dict with dataTables based on simple dictionary"""
#
#    exp_conditions={}
#    for animal in cond_dict:
#        exp_conditions[animal]=pd.DataFrame([cond_dict[animal]], columns=['Cond'])
#    
#    return exp_conditions


test_projects=["test_multi_topview","test_square_arena_topview"]

projects={}
supervised_annotations={}

for test_project in test_projects:

    if test_project=="test_multi_topview":
        arena_detection="circular-autodetect"
        animal_ids=["B","W"]
    else:
        arena_detection="polygonal-manual"
        animal_ids=None

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "plot_examples", test_project),
        video_path=os.path.join(
            ".", "tests", "plot_examples", test_project, "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "plot_examples", test_project, "Tables"
        ),
        arena=arena_detection,
        bodypart_graph="deepof_11",
        video_scale=380,
        animal_ids=animal_ids,
        video_format=".mp4",
        table_format=".h5",
    ).create(force=True, test=True)

    cond_dict={'test': 'odd' ,'test2': 'even','test3': 'odd' ,'test4': 'even','test5': 'odd' ,'test6': 'even'}
    for animal in cond_dict:
        cond_dict[animal]=pd.DataFrame([cond_dict[animal]], columns=['Cond'])

    prun._exp_conditions=cond_dict
    supervised = prun.supervised_annotation()

    projects[test_project]=prun
    supervised_annotations[test_project]=supervised


class plot_info:
    """
    Class to save relevant plot information from a Matplotlib plot object.
    """
    
    def __init__(self):

        self.plot_info={}
        self.ignore=[]

    
    def store(self, plt_obj):

        plot_info = {}
        N_axes = len(plt_obj.gcf().get_axes())  # Get the number of axes (subplots)

        # Iterate through each axis (subplot)
        for k in range(N_axes):
            ax = plt_obj.gcf().get_axes()[k]  # Get the current axis
            plot_info[f'axes_{k}'] = {}  # Initialize a dictionary for each axis

            # Extract lines
            lines = ax.get_lines()
            if lines:
                plot_info[f'axes_{k}']['lines'] = []
                for line in lines:
                    xdata = line.get_xdata()
                    ydata = line.get_ydata()
                    plot_info[f'axes_{k}']['lines'].append({'xdata': xdata, 'ydata': ydata})

            # Extract scatter plots
            scatter_plots = ax.collections
            if not (hasattr(scatter_plots, '__self__') and scatter_plots.__self__ is not None):
                plot_info[f'axes_{k}']['scatter'] = []
                for scatter in scatter_plots:
                    offsets = scatter.get_offsets()
                    xdata = offsets[:, 0]
                    ydata = offsets[:, 1]
                    plot_info[f'axes_{k}']['scatter'].append({'xdata': xdata, 'ydata': ydata})

            # Extract bar containers
            bar_containers = ax.containers
            if isinstance(bar_containers, list):
                plot_info[f'axes_{k}']['bars'] = []
                for bars in bar_containers:
                    bar_data = []
                    for bar in bars:
                        bar_data.append({'x': bar.get_x(), 'height': bar.get_height()})
                    plot_info[f'axes_{k}']['bars'].append(bar_data)

            # Extract error bar objects
            error_bars = ax.errorbar
            if not (hasattr(error_bars, '__self__') and error_bars.__self__ is not None): #check if method
                for error in error_bars:
                    xdata = error[0].get_xdata()
                    ydata = error[0].get_ydata()
                    yerr = error[1].get_ydata() if len(error) > 1 else None
                    plot_info[f'axes_{k}']['error_bars'] = {'xdata': xdata, 'ydata': ydata, 'yerr': yerr}

            # Extract histogram data
            histograms = ax.patches  # Histograms are stored as patches
            if not (hasattr(histograms, '__self__') and histograms.__self__ is not None): #check if method:
                plot_info[f'axes_{k}']['histograms'] = []
                for hist in histograms:
                    if isinstance(hist,patches.Rectangle):
                        height = hist.get_height()
                        x = hist.get_x()
                        width = hist.get_width()
                        plot_info[f'axes_{k}']['histograms'].append({'x': x, 'height': height, 'width': width})
                    elif isinstance(hist,patches.Patch):
                        vertices = hist.get_verts()
                        plot_info[f'axes_{k}']['histograms'].append({'vertices': vertices})

            #extract significance data
            first_annot=True
            for artist in ax.get_children():
                if isinstance(artist, plt.Annotation) or isinstance(artist, plt.Text):
                    if first_annot:
                        plot_info[f'axes_{k}']['annotations']=[]
                        first_annot=False
                    plot_info[f'axes_{k}']['annotations'].append({'text': artist.get_text(), 'position': artist.get_position()})



        self.plot_info=plot_info

    def save(self, path):

        with open(path, 'wb') as f:
            pickle.dump(self.plot_info, f)

    def load(self, path):

        with open(path, 'rb') as f:
            self.plot_info = pickle.load(f)
        

    def compare(self, plt_obj2):
        """start of recursive pltot object comparison.

        Args:
            value1 (any): first thing
            value2 (any): second thing
        """

        path=''
        return self._deep_dict_compare(self.plot_info, plt_obj2.plot_info, path=path)

    
    def _deep_dict_compare(self, dict1, dict2, path):
        """Recursive comparison of dictionaries

        Args:
            dict1 (dict): first dictionary
            dict2 (dict): second dictionary
            path (str): path so into the object, keeps track of superordinate dictionaries and lists this dictionary may be part of. Can be used to exclude sections of the object from comparisons
        """

        if dict1.keys() != dict2.keys():
            return (False, path)
        
        for key in dict1.keys():
            path_new=path+'['+key+']'
            if not (path_new in self.ignore):
                value1, value2 = dict1[key], dict2[key]
                out = self._deep_compare(value1,value2, path_new)
                if not out[0]:
                    return out
        return (True, '')


    def _deep_list_compare(self, list1,list2, path):
        """Recursive comparison of lists

        Args:
            list1 (dict): first list
            list2 (dict): second list
            path (str): path so into the object, keeps track of superordinate dictionaries and lists this dictionary may be part of. Can be used to exclude sections of the object from comparisons
        """

        if len(list1) != len(list2):
            return (False, path)
        
        for k in range(0,len(list1)):
            path_new=path+'['+str(k)+']'
            if not (path_new in self.ignore):
                value1, value2 = list1[k], list2[k]
                out = self._deep_compare(value1,value2, path_new)
                if not out[0]:
                    return out
        return (True, '')        

    def _deep_compare(self,value1,value2,path):
        """part of recursive object comparison. Goes deeper if the value is a dict or list, otherwise compares entry

        Args:
            value1 (any): first thing
            value2 (any): second thing
            path (str): path so into the object, keeps track of superordinate dictionaries and lists this dictionary may be part of. Can be used to exclude sections of the object from comparisons
        """

        if isinstance(value1, dict) and isinstance(value2, dict):
            out = self._deep_dict_compare(value1, value2, path)
            if not out[0]:
                return out
                
        elif isinstance(value1, list) and isinstance(value2, list):
            out = self._deep_list_compare(value1, value2, path)
            if not out[0]:
                return out
                
        elif isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
            if value1.shape !=value2.shape:
                return (False,path)
            
            if (hasattr(value1,'mask') and hasattr(value2,'mask')
                and not all(value1.mask==value2.mask)):
                return (False,path)
                
            if (np.abs(np.array(value1)-np.array(value2)) > 0.01).any():
                return (False,path)  

        elif isinstance(value1, str) and isinstance(value2, str):
            if value1 != value2:
                return (False,path)
            
        elif isinstance(value1, tuple) and isinstance(value2, tuple):
            if (np.abs(np.array(value1)-np.array(value2)) > 0.01).any():
                return (False,path)
                
        else:
            if np.abs(value1-value2) > 0.01:
                return (False,path)

        
        return (True,'') 
  

###############################################
# Test plot-functions based on saved examples #
###############################################


def test_plot_gantt(update=False):

    _init_plot_environment(default=True)

    fig_ref=os.path.join(
        ".", "tests", "plot_examples", "Plots", "plot_gantt.pkl"
    )  
        
    key_1="test"
    signal_overlay=pd.Series(np.sin(np.linspace(0, 200 * np.pi, supervised_annotations["test_multi_topview"][key_1].shape[0])))


    np.random.seed(42)
    # Create a 2 x soft_counts array filled with zeros
    data = np.zeros((5, supervised_annotations["test_multi_topview"][key_1].shape[0]))
    # Randomly set some values to 1 with a low probability
    data.ravel()[np.random.rand(len(data.ravel())) < 0.01] = 1
    # Create a pandas DataFrame from the array
    additional_checkpoints = pd.DataFrame(data)


    fig, ([ax1,ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(3, 2, figsize=(20, 20))


    deepof.visuals.plot_gantt(
        projects["test_multi_topview"],
        key_1,
        bin_index=1,
        bin_size=10,
        supervised_annotations=supervised_annotations["test_multi_topview"],
        signal_overlay=signal_overlay,
        additional_checkpoints=additional_checkpoints,
        ax=ax1
    )

    deepof.visuals.plot_gantt(
        projects["test_multi_topview"],
        key_1,
        bin_index="0:0:10",
        bin_size="0:0:10",
        supervised_annotations=supervised_annotations["test_multi_topview"],
        signal_overlay=signal_overlay,
        ax=ax2
    )

    deepof.visuals.plot_gantt(
        projects["test_square_arena_topview"],
        key_1,
        supervised_annotations=supervised_annotations["test_square_arena_topview"],
        ax=ax3
    )

    deepof.visuals.plot_gantt(
        projects["test_square_arena_topview"],
        key_1,
        bin_index="0:0:5.5",
        bin_size="0:0:1.5",
        supervised_annotations=supervised_annotations["test_square_arena_topview"],
        behaviors_to_plot=["sniffing","climbing","lookaround"],
        ax=ax4
    )

    deepof.visuals.plot_gantt(
        projects["test_multi_topview"],
        key_1,
        bin_index="0:0:0",
        bin_size="0:0:25",
        supervised_annotations=supervised_annotations["test_multi_topview"],
        behaviors_to_plot=["B_lookaround","B_climbing","name_error"],
        ax=ax5
    )

    deepof.visuals.plot_gantt(
        projects["test_multi_topview"],
        key_1,
        supervised_annotations=supervised_annotations["test_multi_topview"],
        signal_overlay=signal_overlay,
        additional_checkpoints=additional_checkpoints,
        behaviors_to_plot=["W_lookaround"],
        ax=ax6
    )

    plt.tight_layout()
    
    _compare_plots(plt=plt, fig_ref=fig_ref, ignore=[''], update=update)


def test_plot_enrichment(update=False):

    _init_plot_environment(default=True)

    fig_ref=os.path.join(
        ".", "tests", "plot_examples", "Plots", "plot_enrichment.pkl"
    ) 


    fig, ([ax1,ax2], [ax3, ax4], [ax5, ax6], [ax7, ax8]) = plt.subplots(4, 2, figsize=(20, 20))

    deepof.visuals.plot_enrichment(
        projects["test_multi_topview"],
        supervised_annotations=supervised_annotations["test_multi_topview"],
        add_stats="Mann-Whitney",
        bin_size= 20,
        bin_index= 0,
        plot_speed=True,
        ax = ax1,
        exp_condition="Cond",
    )

    deepof.visuals.plot_enrichment(
        projects["test_multi_topview"],
        supervised_annotations=supervised_annotations["test_multi_topview"],
        add_stats="Mann-Whitney",
        bin_size= "00:20:00",
        bin_index= "00:00:00",
        polar_depiction=True,
        plot_speed=True,
        ax = ax2,
        exp_condition="Cond",
    )
    deepof.visuals.plot_enrichment(
        projects["test_square_arena_topview"],
        supervised_annotations=supervised_annotations["test_square_arena_topview"],
        add_stats="Mann-Whitney",
        plot_speed=False,
        normalize=True,
        bin_size= "Blubb",
        bin_index= "00:01:00",
        ax = ax3,
        exp_condition="Cond",
    )
    deepof.visuals.plot_enrichment(
        projects["test_square_arena_topview"],
        supervised_annotations=supervised_annotations["test_square_arena_topview"],
        add_stats="Mann-Whitney",
        ax = ax4,
        polar_depiction=True,
        exp_condition="Cond",
    )
    deepof.visuals.plot_enrichment(
        projects["test_multi_topview"],
        supervised_annotations=supervised_annotations["test_multi_topview"],
        plot_speed=False,
        add_stats="Mann-Whitney",
        exp_condition="Cond",
        exp_condition_order=['even', 'odd'],
        normalize=True,
        ax = ax5,
    )

    deepof.visuals.plot_enrichment(
        projects["test_multi_topview"],
        supervised_annotations=supervised_annotations["test_multi_topview"],
        polar_depiction=True,
        add_stats="Mann-Whitney",
        exp_condition="Cond",
        exp_condition_order=['even', 'odd'],
        normalize=False,
        ax = ax6,
    )

    deepof.visuals.plot_enrichment(
        projects["test_multi_topview"],
        supervised_annotations=supervised_annotations["test_multi_topview"],
        plot_speed=False,
        add_stats=False,
        exp_condition="Cond",
        precomputed_bins= ([True] * 10),
        exp_condition_order=['even', 'odd'],
        normalize=True,
        ax = ax7,
    )

    deepof.visuals.plot_enrichment(
        projects["test_multi_topview"],
        supervised_annotations=supervised_annotations["test_multi_topview"],
        polar_depiction=True,
        add_stats=False,
        bin_size= "0:1:00",
        bin_index= "00:0:10",
        precomputed_bins= ([True] * 10),
        exp_condition="Cond",
        exp_condition_order=['even', 'odd'],
        normalize=False,
        ax = ax8,
    )


    plt.tight_layout()
    
    ignore=['[axes_0][lines]','[axes_1][lines]','[axes_2][lines]','[axes_3][lines]'] 
    _compare_plots(plt=plt, fig_ref=fig_ref, ignore=ignore, update=update)


def test_plot_embeddings(update=False):

    _init_plot_environment(default=True)

    fig_ref=os.path.join(
        ".", "tests", "plot_examples", "Plots", "plot_embeddings.pkl"
    ) 


    fig, ([ax1,ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(20, 20))

    deepof.visuals.plot_embeddings(
        projects["test_multi_topview"],
        supervised_annotations=supervised_annotations["test_multi_topview"],
        aggregate_experiments="mean",
        bin_size=20,
        bin_index=0,
        add_stats = "Mann-Whitney",
        verbose = False,
        exp_condition = 'Cond',
        ax = ax1
    )

    deepof.visuals.plot_embeddings(
        projects["test_multi_topview"],
        supervised_annotations=supervised_annotations["test_multi_topview"],
        aggregate_experiments="median",
        bin_size="0:0:20",
        bin_index="0:0:0",
        add_stats = "Mann-Whitney",
        verbose = False,
        exp_condition = 'Cond',
        ax = ax2
    )
    deepof.visuals.plot_embeddings(
        projects["test_multi_topview"],
        supervised_annotations=supervised_annotations["test_multi_topview"],
        aggregate_experiments="mean",
        normative_model = 'even',
        add_stats = "Mann-Whitney",
        verbose = False,
        exp_condition = 'Cond',
        ax = [ax3, ax4],
    )  

    plt.tight_layout()

    _compare_plots(plt=plt, fig_ref=fig_ref, ignore=[''], update=update)


def test_plot_heatmaps(update=False):

    _init_plot_environment(default=True)

    fig_ref=os.path.join(
        ".", "tests", "plot_examples", "Plots", "plot_heatmaps.pkl"
    ) 


    fig, ([ax1,ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(3, 2, figsize=(20, 20))

    deepof.visuals.plot_heatmaps(
        projects["test_multi_topview"],
        "B_Nose",
        center="arena",
        exp_condition="Cond",
        condition_value='odd',
        bin_size=20,
        bin_index=0,
        display_arena=True,
        experiment_id="test",
        show=False,
        ax = ax1
    )
    deepof.visuals.plot_heatmaps(
        projects["test_multi_topview"],
        "B_Nose",
        center="arena",
        exp_condition="Cond",
        condition_value='odd',
        bin_size="0:0:20",
        bin_index="0:0:0",
        display_arena=True,
        experiment_id="test",
        show=False,
        ax = ax2
    )

    deepof.visuals.plot_heatmaps(
        projects["test_multi_topview"],
        "W_Nose",
        center="arena",
        exp_condition="Cond",
        condition_value='even',
        display_arena=False,
        show=False,
        ax = ax3
    )
    deepof.visuals.plot_heatmaps(
        projects["test_square_arena_topview"],
        "Spine_2",
        center="arena",
        bin_size=20,
        bin_index=0,
        display_arena=True,
        experiment_id="test3",
        show=False,
        ax = ax4,
    ) 
    deepof.visuals.plot_heatmaps(
        projects["test_square_arena_topview"],
        "Nose",
        center="arena",
        display_arena=True,
        show=False,
        ax = ax5,
    ) 
    deepof.visuals.plot_heatmaps(
        projects["test_square_arena_topview"],
        "Nose",
        center="arena",
        bin_size=10,
        bin_index=0,
        display_arena=True,
        experiment_id="test",
        show=False,
        ax = ax6,
    ) 


    plt.tight_layout()

    
    _compare_plots(plt=plt, fig_ref=fig_ref, ignore=[''], update=update)


def test_plot_behavior_trends(update=False):

    _init_plot_environment(default=True)

    fig_ref=os.path.join(
        ".", "tests", "plot_examples", "Plots", "plot_behavior_trends.pkl"
    )    

    custom_time_bins = [[0, 124], [125, 249], [250, 374], [375, 499], [500, 624], [625, 749]]

    custom_time_binst = [
        ['0:0:0', '0:0:5'], 
        ['0:0:5.05', '0:0:10'], 
        ['0:0:10.05', '0:0:15'], 
        ['0:0:15.05', '0:0:20'], 
        ['0:0:20.05', '0:0:25'], 
        ['0:0:25.05', '0:0:30']
    ]

    fig, ([ax1,ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(3, 2, figsize=(20, 20))

    deepof.visuals.plot_behavior_trends(
        projects["test_multi_topview"],
        supervised_annotations=supervised_annotations["test_multi_topview"],
        behavior_to_plot='B_lookaround',
        polar_depiction=False,
        show_histogram=True,
        normalize=False,
        ax=ax1,
        custom_time_bins=custom_time_bins,
    )

    deepof.visuals.plot_behavior_trends(
        projects["test_multi_topview"],
        supervised_annotations=supervised_annotations["test_multi_topview"],
        behavior_to_plot='B_lookaround',
        polar_depiction=True,
        show_histogram=False,
        normalize=False,
        ax=ax2,
        custom_time_bins=custom_time_binst,
    )

    deepof.visuals.plot_behavior_trends(
        projects["test_square_arena_topview"],
        supervised_annotations=supervised_annotations["test_square_arena_topview"],
        behavior_to_plot='huddle',
        polar_depiction=False,
        normalize=False,
        N_time_bins=4,
        ax=ax3,
    )

    deepof.visuals.plot_behavior_trends(
        projects["test_square_arena_topview"],
        supervised_annotations=supervised_annotations["test_square_arena_topview"],
        behavior_to_plot='huddle',
        polar_depiction=True,
        normalize=True,
        N_time_bins=4,
        ax=ax4,
    )

    deepof.visuals.plot_behavior_trends(
        projects["test_multi_topview"],
        supervised_annotations=supervised_annotations["test_multi_topview"],
        behavior_to_plot='W_speed',
        normalize=True,
        polar_depiction=False,
        custom_time_bins=custom_time_binst,
        hide_time_bins=[False,False,False]+[True]+[False,False],
        ax=ax5,
    )

    deepof.visuals.plot_behavior_trends(
        projects["test_multi_topview"],
        supervised_annotations=supervised_annotations["test_multi_topview"],
        behavior_to_plot='W_speed',
        normalize=True,
        polar_depiction=True,
        custom_time_bins=custom_time_bins,
        hide_time_bins=[False,False,False]+[True]+[False,False],
        ax=ax6,
    )

    plt.tight_layout()

    _compare_plots(plt=plt, fig_ref=fig_ref, ignore=[''], update=update)



###############################################
#  Clean up (remove created project folders)  #
###############################################

def cleanup():

    for k in range(0,len(test_projects)):
        rmtree(
            os.path.join(
                ".", "tests", "plot_examples", test_projects[k], "deepof_project"
            )
        )

###############################################
#  Helper functions (to avoid redunant code)  #
###############################################

def _compare_plots(plt=None, fig_ref='', ignore=[''], update=False):

    #just do nothing when this function is called by automaized testing
    if plt is None:
        return None

    fig_out= fig_ref.rsplit(".pkl", 1)[0] + ".png"
    fig_comp = fig_ref.rsplit(".pkl", 1)[0] + "_comp.png"

    if update:
        plt.savefig(fig_out)
        plt_info_ref=plot_info()
        plt_info_ref.store(plt)
        plt_info_ref.save(fig_ref)
    else:
        plt.savefig(fig_comp)
        plt_info_ref=plot_info()
        plt_info_ref.load(fig_ref)
        plt_info=plot_info()
        plt_info.store(plt)
        #will ignore variation line and significance comparison lines for bar plots (but not presnce / absence of '*', 'ns' and the like)
        plt_info.ignore=ignore

        out = plt_info.compare(plt_info_ref)
        print(out[1])
        assert out[0]


def _init_plot_environment(default=False):
    """
    Sets global parameters for Matplotlib and Seaborn to ensure consistent plotting aesthetics.
    """

    if not default:
        return None

    use_backend('Agg')

    # Set default figure size
    plt.rcParams['figure.figsize'] = (20, 20)
    
    # Set font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # Set line width
    plt.rcParams['lines.linewidth'] = 2
    
    # Set grid style
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5  # Set grid transparency
    
    # Set Seaborn style
    sns.set_style("whitegrid")
    
    # Set Seaborn color palette
    sns.set_palette("deep")
    