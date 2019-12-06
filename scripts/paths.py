# simple container for the paths to the single checkpoints; contains all
# experiments considered for the thesis and more

def get_paths(date):
    paths = {}

    # 2600x1300, bs=1, lr=1e-4, unsegmented
    paths["19-04-17_16-03-15"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/ddsm_wi_mass_IN_2600x1300/" \
                                 "19-04-17_16-03-15/checkpoints/run_00"

    # 2600x1300, bs=1, lr=1e-5, unsegmented
    paths["19-04-17_16-05-50"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_2600x1300_lr_1e-5/" \
                                 "19-04-17_16-05-50/checkpoints/run_00"

    # 2600x1300, bs=1, lr=1e-6, unsegmented
    paths["19-04-17_16-06-24"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_2600x1300_lr_1e-6/" \
                                 "19-04-17_16-06-24/checkpoints/run_00"

    # 2600x1300, bs=1, lr=1e-4, segmented
    paths["19-04-17_21-36-36"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_2600x1300_lr_1e-4_bs_1_seg/"\
                                 "19-04-17_21-36-36/checkpoints/run_00"

    # 1800x900, bs=1, lr=1e-4, unsegmented
    paths["19-04-17_21-41-08"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_1800x900_lr_1e-4_bs_1/" \
                                 "19-04-17_21-41-08/checkpoints/run_00"

    # 1800x900, bs=1, lr=1e-4, segmented
    paths["19-04-17_21-35-44"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_1800x900_lr_1e-4_bs_1_seg/" \
                                 "19-04-17_21-35-44/checkpoints/run_00"

    # 1800x900, bs=2, lr=1e-4, segmented
    paths["19-04-17_21-34-16"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_1800x900_lr_1e-4_bs_2_seg/" \
                                 "19-04-17_21-34-16/checkpoints/run_00"

    # 1300x650, bs=1, lr=1e-4, unsegmented
    paths["19-04-17_21-23-22"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_1300x650_lr_1e-4_bs_1/" \
                                 "19-04-17_21-23-22/checkpoints/run_00"

    # 1300x650, bs=1, lr=1e-4, segmented
    paths["19-04-19_21-57-44"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_1300x650_lr_1e-4_bs_1_seg/" \
                                 "19-04-19_21-57-44/checkpoints/run_00"

    # 1300x650, bs=2, lr=1e-4, segmented
    paths["19-04-17_21-28-44"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_1300x650_lr_1e-4_bs_2_seg/" \
                                 "19-04-17_21-28-44/checkpoints/run_00"

    #-------------------------------------------------------------------------

    # RN152, IN, 1800x900, bs=1, lr=1e-5, unsegmented
    paths["19-04-18_20-26-01"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN152_1800x900/" \
                                 "19-04-18_20-26-01/checkpoints/run_00"

    # RN101, IN, 1800x900, bs=1, lr=1e-5, unsegmented
    paths["19-04-18_19-28-15"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN101_1800x900/" \
                                 "19-04-18_19-28-15/checkpoints/run_00"

    # RN50, IN, 1800x900, bs=1, lr=1e-5, unsegmented
    paths["19-04-18_16-28-02"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1800x900/" \
                                 "19-04-18_16-28-02/checkpoints/run_00"

    # RN34, IN, 1800x900, bs=1, lr=1e-5, unsegmented
    paths["19-04-18_20-29-55"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN34_1800x900/" \
                                 "19-04-18_20-29-55/checkpoints/run_00"

    # RN18, IN, 1800x900, bs=1, lr=1e-5, unsegmented
    paths["19-04-18_20-31-43"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN18_1800x900/" \
                                 "19-04-18_20-31-43/checkpoints/run_00"

    #------------------------------------------------------------------------

    # RN152, IN, 1800x900, bs=1, lr=1e-6, unsegmented
    paths["19-04-30_19-32-54"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN152_1800x900_lr_1e-6/" \
                                 "19-04-30_19-32-54/checkpoints/run_00"

    # RN152, IN, 1800x900, bs=1, lr=1e-4, unsegmented
    paths["19-05-01_10-38-36"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN152_1800x900_lr_1e-4/" \
                                 "19-05-01_10-38-36/checkpoints/run_00"

    # RN101, IN, 1800x900, bs=1, lr=1e-6, unsegmented
    paths["19-04-30_19-35-09"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN101_1800x900_lr_1e-6/" \
                                 "19-04-30_19-35-09/checkpoints/run_00"

    # RN101, IN, 1800x900, bs=1, lr=1e-4, unsegmented
    paths["19-04-30_19-45-58"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN101_1800x900_lr_1e-4/" \
                                 "19-04-30_19-45-58/checkpoints/run_00"

    # RN34, IN, 1800x900, bs=1, lr=1e-6, unsegmented
    paths["19-04-30_19-38-27"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN34_1800x900_lr_1e-6/" \
                                 "19-04-30_19-38-27/checkpoints/run_00"

    # RN34, IN, 1800x900, bs=1, lr=1e-4, unsegmented
    paths["19-04-30_19-53-14"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN34_1800x900_lr_1e-4/" \
                                 "19-04-30_19-53-14/checkpoints/run_00"

    # RN18, IN, 1800x900, bs=1, lr=1e-6, unsegmented
    paths["19-04-30_19-39-47"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN18_1800x900_lr_1e-6/" \
                                 "19-04-30_19-39-47/checkpoints/run_00"

    # RN18, IN, 1800x900, bs=1, lr=1e-4, unsegmented
    paths["19-04-30_19-47-44"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN18_1800x900_lr_1e-4/" \
                                 "19-04-30_19-47-44/checkpoints/run_00"

    #-------------------------------------------------------------------------
    # RN152, IN, 1300x650, bs=1, lr=1e-5, unsegmented
    paths["19-04-19_13-58-22"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN152_1300x650/" \
                                 "19-04-19_13-58-22/checkpoints/run_00"

    # RN101, IN, 1300x650, bs=1, lr=1e-5, unsegmented
    paths["19-04-19_14-00-07"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN101_1300x650/" \
                                 "19-04-19_14-00-07/checkpoints/run_00"

    # RN50, IN, 1300x650, bs=1, lr=1e-5, unsegmented
    paths["19-04-18_16-47-31"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1300x650/" \
                                 "19-04-18_16-47-31/checkpoints/run_00"

    # RN34, IN, 1300x650, bs=1, lr=1e-5, unsegmented
    paths["19-04-18_20-35-16"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN34_1300x650/" \
                                 "19-04-18_20-35-16/checkpoints/run_00"

    # RN18, IN, 1300x650, bs=1, lr=1e-5, unsegmented
    paths["19-04-18_20-33-44"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN18_1300x650/" \
                                 "19-04-18_20-33-44/checkpoints/run_00"

    #------------------------------------------------------------------------

    # RN50, IN, 1300x650, bs=1, lr=1e-5, unsegmented, sched.
    paths["19-04-19_21-48-21"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1300x650_sched/" \
                                 "19-04-19_21-48-21/checkpoints/run_00"

    # RN34, IN, 1300x650, bs=1, lr=1e-5, unsegmented, sched.
    paths["19-04-19_22-03-11"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN34_1300x650_sched/" \
                                 "19-04-19_22-03-11/checkpoints/run_00"

    # RN18, IN, 1300x650, bs=1, lr=1e-5, unsegmented, sched.
    paths["19-04-19_22-04-44"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN18_1300x650_sched/" \
                                 "19-04-19_22-04-44/checkpoints/run_00"

    #------------------------------------------------------------------------

    # RN18, IN, 2600x1300, bs=1, lr=1e-5, unsegmented
    paths["19-04-19_16-12-30"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN18_2600x1300/" \
                                 "19-04-19_16-12-30/checkpoints/run_00"

    # RN18, IN, 1800x900, bs=1, lr=1e-5, segmented
    paths["19-04-19_18-17-01"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN18_1800x900_seg/" \
                                 "19-04-19_18-17-01/checkpoints/run_00"

    # RN18, IN, 4400x2200, bs=1, lr=1e-5, segmented
    paths["19-04-19_17-42-06"] = "/home/temp/moriz/checkpoints/retinanet/test/" \
                                 "test/" \
                                 "19-04-19_17-42-06/checkpoints/run_00"

    #------------------------------------------------------------------------
    # RN18, IN, 900x900, bs=1, lr=1e-5, crops
    paths["19-04-21_20-44-17"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_900x900_bs_1/" \
                                 "19-04-21_20-44-17/checkpoints/run_00"

    # RN34, IN, 900x900, bs=1, lr=1e-5, crops
    paths["19-04-24_12-29-08"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN34_900x900_lr_1e-5_bs_1/" \
                                 "19-04-24_12-29-08/checkpoints/run_00"

    # RN50, IN, 600x600, bs=1, lr=1e-5, crops
    paths["19-04-19_12-59-19"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN50_600x600/" \
                                 "19-04-19_12-59-19/checkpoints/run_00"

    # RN50, IN, 900x900, bs=1, lr=1e-5, crops, broken
    paths["19-04-19_13-01-54"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN50_900x900/" \
                                 "19-04-19_13-01-54/checkpoints/run_00"

    # RN50, IN, 1200x1200, bs=1, lr=1e-5, crops
    paths["19-04-19_13-02-56"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN50_1200x1200/" \
                                 "19-04-19_13-02-56/checkpoints/run_00"

    # RN50, IN, 900x900, bs=2, lr=1e-5, crops
    paths["19-04-21_20-42-38"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN50_900x900_bs_2/" \
                                 "19-04-21_20-42-38/checkpoints/run_00"

    # RN50, IN, 900x900, bs=4, lr=1e-5, crops
    paths["19-04-21_20-41-55"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN50_900x900_bs_4/" \
                                 "19-04-21_20-41-55/checkpoints/run_00"

    # RN50, IN, 900x900, bs=1, lr=1e-5, crops, better aug
    paths["19-04-21_21-16-26"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN50_900x900_better_aug/" \
                                 "19-04-21_21-16-26/checkpoints/run_00"

    # RN50, IN, 900x900, bs=1, lr=1e-4, crops
    paths["19-04-22_21-55-18"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN50_900x900_lr_1e-4/" \
                                 "19-04-22_21-55-18/checkpoints/run_00"

    # RN50, IN, 900x900, bs=1, lr=1e-4, crops, better aug
    paths["19-04-22_21-54-47"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN50_900x900_lr_1e-4_better_aug/" \
                                 "19-04-22_21-54-47/checkpoints/run_00"

    # RN50, IN, 900x900, bs=1, lr=1e-6, crops
    paths["19-04-22_21-52-34"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN50_900x900_lr_1e-6/" \
                                 "19-04-22_21-52-34/checkpoints/run_00"

    # RN50, IN, 900x900, bs=1, lr=1e-6, crops, better aug
    paths["19-04-22_21-53-52"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN50_900x900_lr_1e-6_better_aug/" \
                                 "19-04-22_21-53-52/checkpoints/run_00"

    # RN50, IN, 900x900, bs=1, lr=1e-5, crops,
    paths["19-04-24_12-28-33"] = "/work/scratch/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN50_900x900_lr_1e-5_bs_1/" \
                                 "19-04-24_12-28-33/checkpoints/run_00"

    #-----------------------------------------------------------------------

    # RN50, IN, 1800x900, bs=1, lr=1e-6, unsegmented
    paths["19-04-20_20-57-44"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1800x900_lr_1e-6_bs_1/" \
                                 "19-04-20_20-57-44/checkpoints/run_00"

    # RN50, IN, 1300x650, bs=1, lr=1e-6, unsegmented
    paths["19-04-20_20-59-20"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1300x650_lr_1e-6_bs_1/" \
                                 "19-04-20_20-59-20/checkpoints/run_00"

    #-------------------------------------------------------------------------

    # RN50, IN, 1800x900, bs=2, lr=1e-4, unsegmented
    paths["19-04-20_21-11-42"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1800x900_lr_1e-4_bs_2/" \
                                 "19-04-20_21-11-42/checkpoints/run_00"

    # RN50, IN, 1800x900, bs=2, lr=1e-5, unsegmented
    paths["19-04-20_19-28-56"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1800x900_bs_2/" \
                                 "19-04-20_19-28-56/checkpoints/run_00"

    # RN50, IN, 1800x900, bs=2, lr=1e-6, unsegmented
    paths["19-04-20_21-28-12"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1800x900_lr_1e-6_bs_2/" \
                                 "19-04-20_21-28-12/checkpoints/run_00"

    #------------------------------------------------------------------------

    # RN50, IN, 1300x650, bs=2, lr=1e-4, unsegmented
    paths["19-04-19_21-56-40"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_1300x650_lr_1e-4_bs_2/" \
                                 "19-04-19_21-56-40/checkpoints/run_00"

    # RN50, IN, 1300x650, bs=2, lr=1e-5, unsegmented
    paths["19-04-20_21-01-16"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1300x650_lr_1e-5_bs_2/" \
                                 "19-04-20_21-01-16/checkpoints/run_00"

    # RN50, IN, 1300x650, bs=2, lr=1e-6, unsegmented
    paths["19-04-20_21-34-14"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1300x650_lr_1e-6_bs_2/" \
                                 "19-04-20_21-34-14/checkpoints/run_00"

    #------------------------------------------------------------------------

    # RN50, IN, 1300x650, bs=4, lr=1e-4, unsegmented
    paths["19-04-20_20-55-33"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1300x650_lr_1e-4_bs_4/" \
                                 "19-04-20_20-55-33/checkpoints/run_00"

    # RN50, IN, 1300x650, bs=4, lr=1e-5, unsegmented
    paths["19-04-20_19-30-50"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1300x650_bs_4/" \
                                 "19-04-20_19-30-50/checkpoints/run_00"

    # RN50, IN, 1300x650, bs=4, lr=1e-6, unsegmented
    paths["19-04-20_21-35-56"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1300x650_lr_1e-6_bs_4/" \
                                 "19-04-20_21-35-56/checkpoints/run_00"

    #---------------------------------------------------------------------------

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, patho labels
    paths["19-04-21_21-00-57"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_lr_1e-5_patho/" \
                                 "19-04-21_21-00-57/checkpoints/run_00"

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, birads labels
    paths["19-04-21_21-01-42"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_lr_1e-5_birads/" \
                                 "19-04-21_21-01-42/checkpoints/run_00"

    #-----------------------------------------------------------------------

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, segmented
    paths["19-04-21_20-54-42"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_lr_1e-5_seg/" \
                                 "19-04-21_20-54-42/checkpoints/run_00"

    # RN50, IN, 2600x1300, bs=1, lr=1e-6, segmented
    paths["19-04-21_20-59-01"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_lr_1e-6_seg/" \
                                 "19-04-21_20-59-01/checkpoints/run_00"

    # RN50, IN, 1800x900, bs=1, lr=1e-5, segmented
    paths["19-04-21_21-06-40"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1800x900_lr_1e-5_seg/" \
                                 "19-04-21_21-06-40/checkpoints/run_00"

    # RN50, IN, 1300x650, bs=1, lr=1e-5, segmented
    paths["19-04-21_21-07-38"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1300x650_lr_1e-5_seg/" \
                                 "19-04-21_21-07-38/checkpoints/run_00"

    #----------------------------------------------------------------------

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex. size
    paths["19-04-22_15-32-16"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_flex/" \
                                 "19-04-22_15-32-16/checkpoints/run_00"

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex. size, seg.
    paths["19-04-22_15-32-24"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_flex_seg/" \
                                 "19-04-22_15-32-24/checkpoints/run_00"

    # RN50, IN, 1800x900, bs=1, lr=1e-5, flex. size
    paths["19-04-22_21-41-07"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1800x900_flex/" \
                                 "19-04-22_21-41-07/checkpoints/run_00"

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex. size, seg.
    paths["19-04-22_21-41-46"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1800x900_flex_seg/" \
                                 "19-04-22_21-41-46/checkpoints/run_00"

    # RN50, IN, 1300x650, bs=1, lr=1e-5, flex. size
    paths["19-04-22_21-58-07"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1300x650_flex/" \
                                 "19-04-22_21-58-07/checkpoints/run_00"

    # RN50, IN, 2600x1300, bs=1, lr=1e-5, flex. size, seg.
    paths["19-04-22_21-59-35"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1300x650_flex_seg/" \
                                 "19-04-22_21-59-35/checkpoints/run_00"

    #--------------------------------------------------------------------

    # RN50, IN, bs=1, lr=1e-5, flex. size version 2, 1080Ti
    paths["19-04-23_21-57-17"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_flex_2/" \
                                 "19-04-23_21-57-17/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, flex. size version 2, 1080Ti, seg.
    paths["19-04-23_22-00-54"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_flex_2_seg/" \
                                 "19-04-23_22-00-54/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, flex. size version 2, 980Ti
    paths["19-04-24_18-48-20"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_flex_980Ti/" \
                                 "19-04-24_18-48-20/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, flex. size version 2, 980Ti, seg.
    paths["19-04-24_18-49-34"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_flex_980Ti_seg/" \
                                 "19-04-24_18-49-34/checkpoints/run_00"

    #------------------------------------------------------------------------

    # RN18, IN, 900x900, bs=1, lr=1e-4, crops, new lazy mode
    paths["19-05-02_12-13-43"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_900x900_lr_1e-4/" \
                                 "19-05-02_12-13-43/checkpoints/run_00"

    # RN18, IN, 900x900, bs=1, lr=1e-5, crops, new lazy mode
    paths["19-04-26_18-31-41"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_900x900_bs_1_lr_1e-5/" \
                                 "19-04-26_18-31-41/checkpoints/run_00"

    # RN18, IN, 900x900, bs=1, lr=1e-6, crops, new lazy mode
    paths["19-05-03_23-52-19"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_900x900_bs_1_lr_1e-6/" \
                                 "19-05-03_23-52-19/checkpoints/run_00"

    # RN34, IN, 900x900, bs=1, lr=1e-4, crops, new lazy mode
    paths["19-05-02_12-09-44"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN34_900x900_lr_1e-4/" \
                                 "19-05-02_12-09-44/checkpoints/run_00"

    # RN34, IN, 900x900, bs=1, lr=1e-5, crops, new lazy mode
    paths["19-04-27_16-26-57"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN34_900x900_bs_1_lr_1e-5_lazy/" \
                                 "19-04-27_16-26-57/checkpoints/run_00"

    # RN34, IN, 900x900, bs=1, lr=1e-6, crops, new lazy mode
    paths["19-05-02_12-11-00"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN34_900x900_lr_1e-6/" \
                                 "19-05-02_12-11-00/checkpoints/run_00"

    # RN50, IN, 900x900, bs=1, lr=1e-4, crops, new lazy mode
    paths["19-05-03_00-14-58"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN50_900x900_bs_1_lr_1e-4/" \
                                 "19-05-03_00-14-58/checkpoints/run_00"

    # RN50, IN, 900x900, bs=1, lr=1e-5, crops, new lazy mode
    paths["19-04-27_16-27-35"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN50_900x900_bs_1_lr_1e-5_lazy/" \
                                 "19-04-27_16-27-35/checkpoints/run_00"

    # RN50, IN, 900x900, bs=1, lr=1e-6, crops, new lazy mode
    paths["19-05-03_00-17-30"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN50_900x900_bs_1_lr_1e-6/" \
                                 "19-05-03_00-17-30/checkpoints/run_00"


    #-------------------------------------------------------------------------

    # RN18, IN, 1200x1200, bs=1, lr=1e-5, crops, new lazy mode
    paths["19-04-26_18-34-13"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_1200x1200_bs_1_lr_1e-5/" \
                                 "19-04-26_18-34-13/checkpoints/run_00"

    # RN18, IN, 600x600, bs=1, lr=1e-5, crops, new lazy mode
    paths["19-04-26_18-30-29"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_600x600_bs_1_lr_1e-5/" \
                                 "19-04-26_18-30-29/checkpoints/run_00"

    #------------------------------------------------------------------------

    # RN18, IN, 1200x1200, bs=1, lr=1e-6, crops, new lazy mode
    paths["19-04-27_17-01-16"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_1200x1200_bs_1_lr_1e-6/" \
                                 "19-04-27_17-01-16/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=2, lr=1e-5, crops, new lazy mode
    paths["19-04-27_16-58-38"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_1200x1200_bs_2_lr_1e-5/" \
                                 "19-04-27_16-58-38/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=2, lr=1e-6, crops, new lazy mode
    paths["19-04-27_16-59-57"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_1200x1200_bs_2_lr_1e-6/" \
                                 "19-04-27_16-59-57/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=4, lr=1e-5, crops, new lazy mode
    paths["19-04-27_16-56-50"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_1200x1200_bs_4_lr_1e-5/" \
                                 "19-04-27_16-56-50/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=4, lr=1e-5, crops, new lazy mode
    paths["19-04-29_22-34-07"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_1200x1200_bs_4_lr_1e-5/" \
                                 "19-04-29_22-34-07/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=4, lr=1e-6, crops, new lazy mode
    paths["19-04-27_16-55-51"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_1200x1200_bs_4_lr_1e-6/" \
                                 "19-04-27_16-55-51/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=8, lr=1e-5, crops, new lazy mode
    paths["19-04-27_16-50-45"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_1200x1200_bs_8_lr_1e-5/" \
                                 "19-04-27_16-50-45/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=8, lr=1e-6, crops, new lazy mode
    paths["19-04-27_16-52-23"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_1200x1200_bs_8_lr_1e-6/" \
                                 "19-04-27_16-52-23/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=8, lr=1e-6, crops, new lazy mode
    paths["19-04-29_22-31-29"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_1200x1200_bs_8_lr_1e-6/" \
                                 "19-04-29_22-31-29/checkpoints/run_00"

    # RN18, BN, 1200x1200, bs=8, lr=1e-5, crops, new lazy mode
    paths["19-04-27_23-11-34"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_BN_RN18_1200x1200_bs_8_lr_1e-5/" \
                                 "19-04-27_23-11-34/checkpoints/run_00"

    # RN18, BN, 1200x1200, bs=8, lr=1e-6, crops, new lazy mode
    paths["19-04-27_23-15-01"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_BN_RN18_1200x1200_bs_8_lr_1e-6/" \
                                 "19-04-27_23-15-01/checkpoints/run_00"

    #-------------------------------------------------------------------------

    # RN18, IN, 900x900, bs=1, lr=1e-5, crops, new lazy mode, better aug
    paths["19-05-04_17-11-20"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_900x900_bs_1_lr_1e-5_ba/" \
                                 "19-05-04_17-11-20/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=1, lr=1e-5, crops, new lazy mode, better aug.
    paths["19-04-27_23-00-22"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_1200x1200_bs_1_lr_1e-5_better_aug/" \
                                 "19-04-27_23-00-22/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=4, lr=1e-5, crops, new lazy mode, better aug.
    paths["19-04-29_22-27-26"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_1200x1200_bs_4_lr_1e-5_better_aug/" \
                                 "19-04-29_22-27-26/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=4, lr=1e-5, crops, new lazy mode, better aug.
    paths["19-05-04_17-13-12"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_1200x1200_bs_4_lr_1e-5_ba/" \
                                 "19-05-04_17-13-12/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=1, lr=1e-6, crops, new lazy mode, better aug.
    paths["19-04-27_23-01-58"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_1200x1200_bs_1_lr_1e-6_better_aug/" \
                                 "19-04-27_23-01-58/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=1, lr=1e-6, crops, new lazy mode, better aug.
    paths["19-05-04_18-10-38"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_1200x1200_bs_1_lr_1e-6_ba/" \
                                 "19-05-04_18-10-38/checkpoints/run_00"

    #------------------------------------------------------------------------


    # RN18, IN, 900x900, bs=2, lr=1e-5, crops, new lazy mode
    paths["19-04-26_18-36-27"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_900x900_bs_2_lr_1e-5/" \
                                 "19-04-26_18-36-27/checkpoints/run_00"

    # RN18, IN, 900x900, bs=4, lr=1e-5, crops, new lazy mode
    paths["19-04-26_19-30-10"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_900x900_bs_4_lr_1e-5/" \
                                 "19-04-26_19-30-10/checkpoints/run_00"

    # RN18, BN, 900x900, bs=4, lr=1e-5, crops, new lazy mode
    paths["19-04-26_19-02-14"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_BN_RN18_900x900_bs_4_lr_1e-5/" \
                                 "19-04-26_19-02-14/checkpoints/run_00"

    # RN18, IN, 900x900, bs=8, lr=1e-5, crops, new lazy mode,
    paths["19-05-02_19-11-02"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_IN_RN18_900x900_bs_8_lr_1e-5/" \
                                 "19-05-02_19-11-02/checkpoints/run_00"

    # RN18, BN, 900x900, bs=8, lr=1e-5, crops, new lazy mode, broken
    paths["19-04-26_18-56-05"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_BN_RN18_900x900_bs_8_lr_1e-5/" \
                                 "19-04-26_18-56-05/checkpoints/run_00"

    # RN18, BN, 900x900, bs=8, lr=1e-5, crops, new lazy mode
    paths["19-05-02_19-27-14"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_mass_BN_RN18_900x900_bs_8_lr_1e-5/" \
                                 "19-05-02_19-27-14/checkpoints/run_00"

    #-----------------------------------------------------------------------

    # INbreast, CV, RN50, IN, 2600x1300, bs=1, lr=1e-4,
    paths["19-04-24_18-55-11"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "inbreast/inbreast_final/" \
                                 "inbreast_wi_mass_RN50_cv_2600x1300_lr_1e-4/" \
                                 "19-04-24_18-55-11/checkpoints/"

    # INbreast, CV, RN50, IN, 2600x1300, bs=1, lr=1e-5,
    paths["19-04-23_10-35-57"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "inbreast/inbreast_final/" \
                                 "inbreast_wi_mass_RN50_cv_2600x1300_lr_1e-5/" \
                                 "19-04-23_10-35-57/checkpoints/"

    # INbreast, CV, RN50, IN, 2600x1300, bs=1, lr=1e-4,
    paths["19-04-24_18-54-06"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "inbreast/inbreast_final/" \
                                 "inbreast_wi_mass_RN50_cv_2600x1300_lr_1e-6/" \
                                 "19-04-24_18-54-06/checkpoints/"

    # INbreast, CV, RN50, IN, 2600x1300, bs=1, lr=1e-5, seg.
    paths["19-04-23_10-37-23"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "inbreast/inbreast_final/" \
                                 "inbreast_wi_mass_RN50_cv_2600x1300_lr_1e-5_seg/" \
                                 "19-04-23_10-37-23/checkpoints/"

    #------------------------------------------------------------------------

    # INbreast, CV, RN50, IN, 600x600, bs=1, lr=1e-5
    paths["19-04-23_10-39-22"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "inbreast/inbreast_final/" \
                                 "inbreast_cb_mass_RN50_cv_600x600_lr_1e-5/" \
                                 "19-04-23_10-39-22/checkpoints/"

    # INbreast, CV, RN50, IN, 900x900, bs=1, lr=1e-5
    paths["19-04-23_10-38-45"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "inbreast/inbreast_final/" \
                                 "inbreast_cb_mass_RN50_cv_900x900_lr_1e-5/" \
                                 "19-04-23_10-38-45/checkpoints/"

    # INbreast, CV, RN50, IN, 1200x1200, bs=1, lr=1e-5
    paths["19-04-23_10-39-43"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "inbreast/inbreast_final/" \
                                 "inbreast_cb_mass_RN50_cv_1200x1200_lr_1e-5/" \
                                 "19-04-23_10-39-43/checkpoints/"

    # INbreast, CV, RN50, IN, 900x900, bs=1, lr=1e-4
    paths["19-04-23_11-24-40"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "inbreast/inbreast_final/" \
                                 "inbreast_cb_mass_RN50_cv_900x900_lr_1e-4/" \
                                 "19-04-23_11-24-40/checkpoints/"

    # INbreast, CV, RN50, IN, 900x900, bs=1, lr=1e-6
    paths["19-04-23_11-25-48"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "inbreast/inbreast_final/" \
                                 "inbreast_cb_mass_RN50_cv_900x900_lr_1e-6/" \
                                 "19-04-23_11-25-48/checkpoints/"

    # INbreast, CV, RN18, IN, 600x600, bs=1, lr=1e-5
    paths["19-05-04_18-14-50"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "inbreast_cb_mass_IN_RN18_600x600_bs_1_lr_1e-5/" \
                                 "19-05-04_18-14-50/checkpoints/"

    # INbreast, CV, RN18, IN, 900x900, bs=1, lr=1e-5
    paths["19-04-29_09-58-41"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "inbreast/inbreast_final/" \
                                 "inbreast_cb_mass_IN_RN18_cv_900x900_lr_1e-5/" \
                                 "19-04-29_09-58-41/checkpoints/"

    # INbreast, CV, RN18, IN, 1200x1200, bs=1, lr=1e-5
    paths["19-04-29_09-57-24"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "inbreast/inbreast_final/" \
                                 "inbreast_cb_mass_IN_RN18_cv_1200x1200_lr_1e-5/" \
                                 "19-04-29_09-57-24/checkpoints/"

    #-------------------------------------------------------------------------

    # RN18, IN, bs=1, lr=1e-5, 2600x1300, flex, comparison
    paths["19-05-05_15-25-37"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN18_2600x1300_bs_1_lr_1e-5/" \
                                 "19-05-05_15-25-37/checkpoints/run_00"

    # RN18, IN, bs=1, lr=1e-5, 2600x1300,flex, multi-stage
    paths["19-05-01_11-25-14"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/"\
                                 "ddsm_wi_mass_IN_RN18_2600x1300_flex_lr_1e-5_CL/" \
                                 "19-05-01_11-25-14/checkpoints/run_00"

    # RN18, IN, bs=1, lr=1e-6, 2600x1300, flex, multi-stage
    paths["19-05-01_11-25-55"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN18_2600x1300_flex_lr_1e-6_CL/" \
                                 "19-05-01_11-25-55/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, 2600x1300, flex, multi-stage
    paths["19-05-01_11-22-10"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_flex_lr_1e-5_CL/" \
                                 "19-05-01_11-22-10/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-6, 2600x1300, flex, multi-stage
    paths["19-05-01_11-23-58"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_flex_lr_1e-6_CL/" \
                                 "19-05-01_11-23-58/checkpoints/run_00"

    #------------------------------------------------------------------------

    # RN18, IN, bs=1, lr=1e-5, 2600x1300, comparison
    paths["19-05-05_16-55-51"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN18_2600x1300_bs_1_lr_1e-5/" \
                                 "19-05-05_16-55-51/checkpoints/run_00"

    # RN18, IN, bs=1, lr=1e-5, 2600x1300, multi-stage
    paths["19-05-05_16-43-00"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN18_2600x1300_bs_1_lr_1e-5_CL/" \
                                 "19-05-05_16-43-00/checkpoints/run_00"

    # RN18, IN, bs=1, lr=1e-6, 2600x1300, multi-stage
    paths["19-05-05_16-47-51"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN18_2600x1300_bs_1_lr_1e-6_CL/" \
                                 "19-05-05_16-47-51/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, 2600x1300, multi-stage
    paths["19-05-05_16-51-09"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_bs_1_lr_1e-5_CL/" \
                                 "19-05-05_16-51-09/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-6, 2600x1300, multi-stage
    paths["19-05-05_16-52-08"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_bs_1_lr_1e-6_CL/" \
                                 "19-05-05_16-52-08/checkpoints/run_00"

    #-------------------------------------------------------------------------
    # RN50, IN, bs=1, lr=1e-5, 2600x1300, one side flex, calc.
    paths["19-04-29_09-47-38"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_calc_IN_RN50_2600x1300_flex/" \
                                 "19-04-29_09-47-38/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-4, 2600x1300, fix, calc.
    paths["19-05-01_23-19-03"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_calc_IN_RN50_2600x1300_lr_1e-4/" \
                                 "19-05-01_23-19-03/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, 2600x1300, fix, calc.
    paths["19-05-01_23-18-12"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_calc_IN_RN50_2600x1300_lr_1e-5/" \
                                 "19-05-01_23-18-12/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-6, 2600x1300, fix, calc.
    paths["19-05-01_23-20-26"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_calc_IN_RN50_2600x1300_lr_1e-6/" \
                                 "19-05-01_23-20-26/checkpoints/run_00"

    #-----------------------------------------------------------------------

    # RN18, IN, 600x600, bs=1, lr=1e-5, crops, calc
    paths["19-05-03_00-12-59"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_calc_IN_RN18_600x600_bs_1_lr_1e-5/" \
                                 "19-05-03_00-12-59/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=1, lr=1e-5, crops, calc
    paths["19-05-03_00-10-53"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_cb_calc_IN_RN18_1200x1200_bs_1_lr_1e-5/" \
                                 "19-05-03_00-10-53/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=1, lr=1e-5, crops, calc
    paths["19-05-07_15-04-32"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "inbreast/inbreast_final/" \
                                 "inbreast_cb_calc_IN_RN18_1200x1200_bs_4_lr_1e-5/" \
                                 "19-05-07_15-04-32/checkpoints/run_00"

    # RN18, IN, 1200x1200, bs=1, lr=1e-5, crops, calc (repeated)
    paths["19-05-07_15-07-25"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "inbreast/inbreast_final/" \
                                 "inbreast_cb_calc_IN_RN18_1200x1200_bs_4_lr_1e-5_2/" \
                                 "19-05-07_15-07-25/checkpoints/run_00"

    #------------------------------------------------------------------------
    # LAST IMPROVEMENTS (seg. , aug.)

    # RN50, IN, bs=1, lr=1e-5, 2600x1300, flex, seg. test
    paths["19-05-01_22-26-23"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_flex_seg_test/" \
                                 "19-05-01_22-26-23/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, 1800x900, flex, seg. test
    paths["19-05-01_22-27-06"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1800x900_flex_seg_test/" \
                                 "19-05-01_22-27-06/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, 2600x1300, fix, seg. test
    paths["19-05-01_22-29-19"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_seg_test/" \
                                 "19-05-01_22-29-19/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, 1800x900, fix,  seg. test
    paths["19-05-01_22-28-33"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1800x900_seg_test/" \
                                 "19-05-01_22-28-33/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, 2600x1300, flex, better aug.
    paths["19-05-01_22-34-30"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_flex_ba/" \
                                 "19-05-01_22-34-30/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, 2600x1300, flex, seg. test, better aug.
    paths["19-05-01_22-32-48"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_flex_seg_ba/" \
                                 "19-05-01_22-32-48/checkpoints/run_00"


    # ------------------------------------------------------------------------
    # TESTS

    # RN18, IN, bs=1, lr=1e-5, 1800x900, multi-stage
    paths["19-04-26_20-59-29"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "test/test/" \
                                 "19-04-26_20-59-29/checkpoints/run_00"

    # RN18, IN, bs=1, lr=1e-5, 1800x900, multi-stage
    paths["19-04-26_21-13-41"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "test/test_2/" \
                                 "19-04-26_21-13-41/checkpoints/run_00"

    # RN18, IN, bs=1, lr=1e-5, 1300x650, birads, softmax
    paths["19-04-27_22-49-30"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "test/test_3/" \
                                 "19-04-27_22-49-30/checkpoints/run_00"

    # RN18, IN, bs=1, lr=1e-5, flex size, 980Ti
    paths["19-04-28_13-52-53"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_flex_980Ti_test/" \
                                 "19-04-28_13-52-53/checkpoints/run_00"

    # RN18, IN, bs=1, lr=1e-5, flex size, 1080Ti
    paths["19-04-28_13-54-55"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_flex_1080Ti_test/" \
                                 "19-04-28_13-54-55/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, 2600x1300, one side flex, better aug.
    paths["19-04-28_21-24-09"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_flex_better_aug/" \
                                 "19-04-28_21-24-09/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, 2600x1300, one side flex, mass,
    # different seg. order
    paths["19-04-30_20-28-21"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_flex_lr_1e-5_seg_test/" \
                                 "19-04-30_20-28-21/checkpoints/run_00"

    #-----------------------------------------------------------------------

    # RN50, IN, bs=1, lr=1e-5, 2600x1300, one side flex, patho
    paths["19-05-03_23-55-21"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_flex_patho/" \
                                 "19-05-03_23-55-21/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, 2600x1300, one side flex, birads
    paths["19-05-03_23-58-08"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_flex_birads/" \
                                 "19-05-03_23-58-08/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, 2600x1300, fix, better aug.
    paths["19-05-03_14-44-34"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_bs_1_lr_1e-5_ba/" \
                                 "19-05-03_14-44-34/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, 2600x1300, fix, seg., better aug.
    paths["19-05-03_14-45-20"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_2600x1300_bs_1_lr_1e-5_seg_ba/" \
                                 "19-05-03_14-45-20/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, 1800x900, fix, better aug.
    paths["19-05-03_15-16-15"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1800x900_bs_1_lr_1e-5_ba/" \
                                 "19-05-03_15-16-15/checkpoints/run_00"

    # RN50, IN, bs=1, lr=1e-5, 1800x900, flex, better aug.
    paths["19-05-03_15-11-55"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN50_1800x900_flex_bs_1_lr_1e-5_ba/" \
                                 "19-05-03_15-11-55/checkpoints/run_00"

    # RN18, IN, bs=1, lr=1e-5, 2600x1300, one side flex,
    paths["19-05-05_15-25-37"] = "/home/temp/moriz/checkpoints/retinanet/" \
                                 "ddsm/ddsm_final/" \
                                 "ddsm_wi_mass_IN_RN18_2600x1300_bs_1_lr_1e-5/" \
                                 "19-05-05_15-25-37/checkpoints/run_00"

    if date not in paths.keys():
        raise KeyError("Unknown path!")
    else:
        return paths[date]