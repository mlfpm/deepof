# @author NoCreativeIdeaForGoodUserName
# encoding: utf-8
# module deepof

"""

Testing module for deepof.visuals

"""

import os
from shutil import rmtree
import numpy as np
import pandas as pd
import deepof.data
import deepof.utils
from matplotlib import pyplot as plt
from PIL import Image


import deepof.data
from deepof.visuals import (
    plot_behavior_trends
)

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
        arena_detection="polygonal-autodetect"
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




###############################################
# Test plot-functions based on saved examples #
###############################################


def test_plot_behavior_trends():

    fig_ref=os.path.join(
        ".", "tests", "plot_examples", "Plots", "plot_behavior_trends.png"
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

    # This loop is only here because, for unknown reasons, the first plots after compiling 
    # have a slightly deviating frame color from all following ones
    for _ in range(0,2):

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
    
    fig_out= os.path.join(
        ".", "tests", "plot_examples", "Plots", "plot_behavior_trends_comp.png"
    )

    plt.savefig(fig_out)

    output_image = np.array(Image.open(fig_out))
    reference_image = np.array(Image.open(fig_ref))
    
    if os.path.exists(fig_out):
        os.remove(fig_out)

    assert output_image.shape == reference_image.shape
    assert np.sum(np.abs(output_image.astype(float)-reference_image.astype(float))) < 1000


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