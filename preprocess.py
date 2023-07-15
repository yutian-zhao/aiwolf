import os
import torch
import copy
import random
import torch.nn.functional as F
import numpy as np

# if __name__ == __main__:
MAX_DAY_LENGTH = 14
NAME = 'debug_data' # 'temp_dataset' # "gat2017log15"
NUM_LOGS = 1 # 10000 # 99998 # 10^5-2
ROLES = ["VILLAGER", "SEER", "MEDIUM", "BODYGUARD", "WEREWOLF", "POSSESSED"]
ROLES_DICT = {"VILLAGER":1, "SEER":2, "MEDIUM":3, "BODYGUARD":4, "WEREWOLF":5, "POSSESSED":6}
TOKEN_TYPES = ['status', 'divine', 'whisper', 'guard', 'attackVote',    'attack', 'talk', 'vote', 'execute', 'result']
dir = f"data/{NAME}" # broken file: 398/076, 023/017
role = "VILLAGER"
num_player = 15
num_channel = 8
data = torch.empty(NUM_LOGS, MAX_DAY_LENGTH, num_channel, num_player, num_player)
labels = torch.empty(NUM_LOGS, num_player)
log_count = 0
for root, dirs, files in os.walk(dir, topdown=True):
    dirs.sort()
    for name in sorted(files):
        if name.endswith(".log"):
            with open(os.path.join(root, name), "r") as f:
                log_count += 1
                lines = f.readlines()
                day_checker = 1
                game_end = False
                id = None
                id_role = {}
                have_voted = []
                # role depend
                game_status = torch.empty((0, num_channel, num_player, num_player))
                day_status = torch.zeros((1, num_channel, num_player, num_player))
                prev_day_vote_matrix = torch.zeros((num_player, num_player))
                vote_matrix = torch.zeros((num_player, num_player))
                for line in lines:
                    line = line.strip()
                    tokens = line.split(",")

                    # check validity
                    if int(tokens[0]) == 0:
                        # TODO: parse divine
                        continue
                        print("THIS SHOULD NOT BE PRINTED FOR DAY 0!")
                    if int(tokens[0]) == 1 + day_checker:
                        ## init
                        # print("Day {} done.".format(day_checker))
                        if id:
                            # TODO: parse divine
                            day_status[0, 0, id] = 1
                        else:
                            villagers_id = []
                            for id, role in id_role.items():
                                if role == "VILLAGER":
                                    villagers_id.append(id)
                            # id = random.choice(villagers_id)
                            id = 2
                            day_status[0, 0, id] = 1
                        have_voted = []
                        day_status[0, 3] = copy.deepcopy(prev_day_vote_matrix)
                        prev_day_vote_matrix = copy.deepcopy(vote_matrix)
                        vote_matrix = torch.zeros((num_player, num_player))
                        game_status = torch.cat((game_status, day_status), dim=0)
                        day_checker += 1
                        day_status = torch.zeros((1, num_channel,  num_player, num_player))
                    elif int(tokens[0]) != day_checker:
                        raise Exception("Day check failed! Should be day {} or {}, but got {}.".format(day_checker, day_checker+1, tokens[0]))
                    assert tokens[1] in TOKEN_TYPES
                    assert not game_end

                    if tokens[1] == 'result':
                        game_end = True
                    
                    # parse Alive status
                    elif tokens[1] == 'status':
                        if day_checker == 1:
                            id_role[int(tokens[2])-1] = tokens[3]
                        if tokens[4] == 'ALIVE':
                            day_status[0, 1, [int(tokens[2])-1]] = 1 # broadcasting
                    
                    elif tokens[1] == 'vote':
                        if not int(tokens[2]) in have_voted:
                            have_voted.append(int(tokens[2]))
                            vote_matrix[int(tokens[2])-1, int(tokens[3])-1] = 1
                    
                    elif tokens[1] == 'talk':
                        # Declare
                        if tokens[-1].startswith('VOTE'):
                            vote_target = int(tokens[-1][-3:-1])
                            day_status[0, 7, int(tokens[4])-1, vote_target-1] = 1
                        elif tokens[-1].startswith('IDENTIFIED'):
                            identified_target = int(tokens[-1].split(" ")[1][-3:-1])
                            identified_result = 1 if tokens[-1].split(" ")[-1] == 'HUMAN' else -1
                            day_status[0, 6, int(tokens[4])-1, identified_target-1] = identified_result
                        elif tokens[-1].startswith('DIVINED'):
                            divined_target = int(tokens[-1].split(" ")[1][-3:-1])
                            divined_result = 1 if tokens[-1].split(" ")[-1] == 'HUMAN' else -1
                            day_status[0, 5, int(tokens[4])-1, divined_target-1] = divined_result
                        elif tokens[-1].startswith('ESTIMATE'):
                            estimate_target = int(tokens[-1].split(" ")[1][-3:-1])
                            estimate_result = ROLES_DICT[tokens[-1].split(" ")[-1]]
                            day_status[0, 4, int(tokens[4])-1, estimate_target-1] = estimate_result
                        elif tokens[-1].startswith('COMINGOUT'):
                            co_result = ROLES_DICT[tokens[-1].split(" ")[-1]]
                            day_status[0, 2, int(tokens[4])-1] = co_result
                
                assert game_status.shape[0] <= MAX_DAY_LENGTH
                # game_status = torch.unsqueeze(F.pad(game_status, (0, 0, 0, 0, 0, 0, 0, MAX_DAY_LENGTH-game_status.shape[0]), "constant", 0), 0)

                for i in range(MAX_DAY_LENGTH-game_status.shape[0]):
                    game_status = torch.cat((game_status, torch.zeros((1, num_channel, num_player, num_player))), dim=0)
                # game_status = torch.unsqueeze(game_status, 0)
                # data = torch.cat((data, game_status), dim=0)
                data[log_count-1] = game_status

                label = []
                for i in range(num_player):
                    label.append(ROLES_DICT[id_role[i]])
                label = torch.tensor(label, dtype=torch.int) # unsqueeze
                # labels = torch.cat((labels, label), dim=0)
                labels[log_count-1] = label
                
    print("{} done.".format(root))

assert log_count == NUM_LOGS # 10000 # 99998
print("{} games loaded.".format(len(data)))
# print(data[0][0])
# print(len(data[0][0]))
# torch.save((data, labels), f"data/{NAME}.pt")
with open("debug.log", 'w') as f, np.printoptions(threshold=np.inf):
    f.write(str(np.array(data)))