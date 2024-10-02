##############################################################
## PoseScript                                               ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

# requires at least Python 3.6 (order preserved in dicts)

import os, sys
import torch
import numpy as np

from posecodes import POSECODE_OPERATORS, distance_between_joint_pairs
from captioning_data import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.codebook import *
import options.option_posecode as option_posecode

################################################################################
## UTILS
################################################################################

args = option_posecode.get_args_parser()
if args.dataname == 'kit':
    folder = "KIT-ML"
else:
    folder = 'HumanML3D'
args.out_dir = os.path.join(args.out_dir, folder, 'codes')
os.makedirs(args.out_dir, exist_ok = True)
args.in_dir = os.path.join(args.in_dir, folder, 'new_joints')

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def list_remove_duplicate_preserve_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# (SMPL-H) skeleton (22 main body + 2*15 hands), from https://meshcapade.wiki/SMPL#skeleton-layout
if args.dataname == "t2m":
    ALL_JOINT_NAMES = [
        'pelvis',
        'left_hip',
        'right_hip',
        'spine1',
        'left_knee',
        'right_knee',
        'spine2',
        'left_ankle',
        'right_ankle',
        'spine3',
        'left_foot',
        'right_foot',
        'neck',
        'left_collar',
        'right_collar',
        'head',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_index1',
        'left_index2',
        'left_index3',
        'left_middle1',
        'left_middle2',
        'left_middle3',
        'left_pinky1',
        'left_pinky2',
        'left_pinky3',
        'left_ring1',
        'left_ring2',
        'left_ring3',
        'left_thumb1',
        'left_thumb2',
        'left_thumb3',
        'right_index1',
        'right_index2',
        'right_index3',
        'right_middle1',
        'right_middle2',
        'right_middle3',
        'right_pinky1',
        'right_pinky2',
        'right_pinky3',
        'right_ring1',
        'right_ring2',
        'right_ring3',
        'right_thumb1',
        'right_thumb2',
        'right_thumb3',
    ]
else:
    #KIT
    ALL_JOINT_NAMES = [
        'pelvis',
        'left_hip',
        'right_hip',
        'spine1',
        'left_knee',
        'right_knee',
        'spine2',
        'left_ankle',
        'right_ankle',
        'spine3',
        'left_foot',
        'right_foot',
        'neck',
        'left_collar',
        'right_collar',
        'head',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
    ]

SMPL2KIT = {0:0,1:16,2:11,3:1,4:17,5:12,6:2,7:19,8:14,9:3,10:20,11:15,12:4,13:None,14:None,15:None,16:5,17:8,18:6,19:9,20:7,21:10}

def format_kit(joints):
    formatted = np.zeros((joints.shape[0], 22, 3))
    for i in range(22):
        if SMPL2KIT[i] is None:
            continue
        formatted[:,i,:] = joints[:,SMPL2KIT[i],:]
    return formatted
      
# Joints that are actually useful for the captioning pipeline
VIRTUAL_JOINTS = ["left_hand", "right_hand", "torso"] # ADD_VIRTUAL_JOINT
JOINT_NAMES = ALL_JOINT_NAMES[:22] + ['left_middle2', 'right_middle2'] + VIRTUAL_JOINTS
JOINT_NAMES2ID = {jn:i for i, jn in enumerate(JOINT_NAMES)}

# Interpretation set (interpretations from the posecode operators + (new
# distinct) interpretations from the set of super-posecodes)
# (preserving the order of the posecode operators interpretations, to easily
# convert operator-specific interpretation ids to global interpretation ids,
# using offsets ; as well as the order of super-posecode interpretations, for 
# compatibility accross runs)
INTERPRETATION_SET = flatten_list([p["category_names"] for p in POSECODE_OPERATORS_VALUES.values()])
sp_interpretation_set = [v[1][1] for v in SUPER_POSECODES if v[1][1] not in INTERPRETATION_SET]
INTERPRETATION_SET += list_remove_duplicate_preserve_order(sp_interpretation_set)
INTPTT_NAME2ID = {intptt_name:i for i, intptt_name in enumerate(INTERPRETATION_SET)}

################################################################################
## MAIN
################################################################################

def main(coords, verbose=True, file_id = None):

    if verbose: print("Formating input...")
    # Select & complete joint coordinates (prosthesis phalanxes, virtual joints)
    # Prepare posecode queries
    # (hold all info about posecodes, essentially using ids)
    p_queries = prepare_posecode_queries()
    sp_queries = {} 
    coords = prepare_input(coords)
    
    if verbose: print("Eval & interprete & elect eligible posecodes...")
    # Eval & interprete & elect eligible elementary posecodes
    p_interpretations, p_eligibility = infer_posecodes(coords, p_queries, sp_queries, verbose=verbose)
    
    new_vec = []
    for key in p_interpretations:
        new_vec.append((p_interpretations[key].numpy()))

    new_vec = np.concatenate(new_vec, axis =1)
    new_vec_formatted = np.zeros((new_vec.shape[0],len(id_to_name)))
    for t in range(new_vec.shape[0]):
        for i in vq_to_trans:
            idx = vq_to_trans[i][int(new_vec[t][i])]
            new_vec_formatted[t][idx] = 1
    f = open(args.out_dir+f"/{file_id}","wb")
    np.save(f, new_vec_formatted.astype(bool))
    f.close()
    return
    

################################################################################
## PREPARE INPUT
################################################################################

ALL_JOINT_NAMES2ID = {jn:i for i, jn in enumerate(ALL_JOINT_NAMES)}

def compute_wrist_middle2ndphalanx_distance(coords):
    x = distance_between_joint_pairs([
        [ALL_JOINT_NAMES2ID["left_middle2"], ALL_JOINT_NAMES2ID["left_wrist"]],
        [ALL_JOINT_NAMES2ID["right_middle2"], ALL_JOINT_NAMES2ID["right_wrist"]]], coords)
    return x.mean().item()

def prepare_input(coords):
    """
    Select coordinates for joints of interest, and complete thems with the
    coordinates of virtual joints. If coordinates are provided for the main 22
    joints only, add a prosthesis 2nd phalanx to the middle L&R fingers, in the
    continuity of the forearm.
    
    Args:
        coords (torch.tensor): size (nb of poses, nb of joints, 3), coordinates
            of the different joints, for several poses; with joints being all
            of those defined in ALL_JOINT_NAMES or just the first 22 joints.
    
    Returns:
        (torch.tensor): size (nb of poses, nb of joints, 3), coordinates
            of the different joints, for several poses; with the joints being
            those defined in JOINT_NAMES
    """
    nb_joints = coords.shape[1]
    ### get coords of necessary existing joints
    if nb_joints == 22:
        # add prosthesis phalanxes
        # distance to the wrist
        x = 0.1367 # found by running compute_wrist_middle2ndphalanx_distance on the 52-joint sized coords of a 20k-pose set
        # direction from the wrist (vectors), in the continuity of the forarm
        left_v = coords[:,ALL_JOINT_NAMES2ID["left_wrist"]] - coords[:,ALL_JOINT_NAMES2ID["left_elbow"]]
        right_v = coords[:,ALL_JOINT_NAMES2ID["right_wrist"]] - coords[:,ALL_JOINT_NAMES2ID["right_elbow"]]
        # new phalanx coordinate
        added_j = [x*left_v/torch.linalg.norm(left_v, axis=1).view(-1,1) \
                        + coords[:,ALL_JOINT_NAMES2ID["left_wrist"]],
                    x*right_v/torch.linalg.norm(right_v, axis=1).view(-1,1) \
                        + coords[:,ALL_JOINT_NAMES2ID["right_wrist"]]]
        added_j = [aj.view(-1, 1, 3) for aj in added_j]
        coords = torch.cat([coords] + added_j, axis=1) # concatenate along the joint axis
    if nb_joints >= 52:
        # remove unecessary joints
        keep_joints_indices = [ALL_JOINT_NAMES2ID[jn] for jn in JOINT_NAMES[:-len(VIRTUAL_JOINTS)]]
        coords = coords[:,keep_joints_indices]
    ### add virtual joints
    added_j = [0.5*(coords[:,JOINT_NAMES2ID["left_wrist"]] + coords[:,JOINT_NAMES2ID["left_middle2"]]), # left hand
                0.5*(coords[:,JOINT_NAMES2ID["right_wrist"]] + coords[:,JOINT_NAMES2ID["right_middle2"]]), # right hand
                1/3*(coords[:,JOINT_NAMES2ID["pelvis"]] + coords[:,JOINT_NAMES2ID["neck"]] + coords[:,JOINT_NAMES2ID["spine3"]]), # torso
                # ADD_VIRTUAL_JOINT
                ]
    added_j = [aj.view(-1, 1, 3) for aj in added_j]
    coords = torch.cat([coords] + added_j, axis=1) # concatenate along the joint axis
    return coords


def prepare_posecode_queries():
    """
    Returns a dict with data attached to each kind of posecode, for all
    posecodes of the given kind. One posecode is defined by its kind, joint set
    and interpretation. The joint set does not always carry the name of the body
    part that is actually described by the posecode, and will make it to the
    text. Hence the key 'focus body part'.
    Specifically:
    - the tensor of jointset ids (1 joint set/posecode, with the size of the
        joint set depending on the kind of posecode). The order of the ids might
        matter.
    - the list of acceptable interpretations ids for each jointset (at least 1
        acceptable interpretation/jointset)
    - the list of unskippable interpretations ids for each jointset (possible to
        have empty lists)
    - the list of support-I interpretation ids for each jointset (possible to
        have empty list)
    - the list of support-II interpretation ids for each jointset (possible to
        have empty list)
    - the name of the main focus body part for each jointset
    - the offset to convert the interpretation ids (valid in the scope of the
        considered posecode operator) to global interpretation ids
    """
    posecode_queries = {}
    offset = 0
    all_ids = []
    for posecode_kind, posecode_list in ALL_ELEMENTARY_POSECODES.items():
        # parse information about the different posecodes
        joint_ids = torch.tensor([[JOINT_NAMES2ID[jname] for jname in p[0]]
                                    if type(p[0])!=str else JOINT_NAMES2ID[p[0]]
                                    for p in posecode_list]).view(len(posecode_list), -1)
 
        unique  = set([tuple(set(jid.tolist())) for jid in joint_ids])
        if len(unique) < len(joint_ids):
            print(f'Error in posecode definition of [{posecode_kind} kind]. A posecode should only be defined once. Check unicity of joint sets (considering involved joints in any order). Change interpretations, as well as the focus body parts if necessary, so that the joint set if used only once for this kind of posecode.')
            sys.exit()

        for k in joint_ids.tolist():
            all_ids.extend(k)
        # save posecode information
        posecode_queries[posecode_kind] = {
            "joint_ids": joint_ids,
            "focus_body_part": [p[1] for p in posecode_list],
            "offset": offset,
        }

        offset += len(POSECODE_OPERATORS_VALUES[posecode_kind]['category_names'])

    return posecode_queries

################################################################################
## INFER POSECODES
################################################################################

def infer_posecodes(coords, p_queries, sp_queries, verbose = True):
    
    # init
    p_interpretations = {}
    p_eligibility = {}
    count = {}
    for p_kind, p_operator in POSECODE_OPERATORS.items():
        # evaluate posecodes
        val = p_operator.eval(p_queries[p_kind]["joint_ids"], coords)

        # to represent a bit human subjectivity, slightly randomize the
        # thresholds, or, more conveniently, simply randomize a bit the
        # evaluations: add or subtract up to the maximum authorized random
        # offset to the measured values.
        val += (torch.rand(val.shape)*2-1) * p_operator.random_max_offset

        # interprete the measured values
        p_intptt = p_operator.interprete(val) + p_queries[p_kind]["offset"]

        # infer posecode eligibility for description
        p_elig = torch.zeros(p_intptt.shape)
        p_interpretations[p_kind] = p_intptt  # size (nb of poses, nb of joint sets)
        p_eligibility[p_kind] = p_elig  # size (nb of poses, nb of joint sets)
    
    # Print information about the number of posecodes
    if verbose:
        total_posecodes = 0
        print("Number of posecodes of each kind:")
        for p_kind, p_elig in p_eligibility.items():
            print(f'- {p_kind}: {p_elig.shape[1]}')
            total_posecodes += p_elig.shape[1]
        print(f'Total: {total_posecodes} posecodes.')

    return p_interpretations, p_eligibility

################################################################################
## EXECUTED PART
################################################################################

if __name__ == "__main__" :

    import numpy as np
    import glob
    from tqdm import tqdm
    for filename in tqdm(glob.glob(args.in_dir+"/*.npy")):
        file_id = filename.split("/")[-1]
        f = open(filename,"rb")
        pred_xyz = np.load(f).reshape((-1,22,3))
        f.close()
        scale = 1
        if args.dataname == "kit":
            pred_xyz= format_kit(pred_xyz)
            scale = 1
        
        for frame in range(pred_xyz.shape[0]):
            pred_xyz[frame] = (np.array([[-1,0,0],[0,1,0],[0,0,-1]]) @ pred_xyz[frame].T).T

        pred_xyz = torch.from_numpy(pred_xyz)/scale
        main(pred_xyz, file_id = file_id, verbose=False)