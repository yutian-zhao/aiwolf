import os
import torch
import copy
import random

# if __name__ == __main__:
ROLES = ["VILLAGER", "SEER", "MEDIUM", "BODYGUARD", "WEREWOLF", "POSSESSED"]
ROLES_DICT = {"VILLAGER":1, "SEER":2, "MEDIUM":3, "BODYGUARD":4, "WEREWOLF":5, "POSSESSED":6}
TOKEN_TYPES = ['status', 'divine', 'whisper', 'guard', 'attackVote',    'attack', 'talk', 'vote', 'execute', 'result']
dir = "data/gat2017log15"
role = "VILLAGER"
num_player = 15
num_channel = 8
data = []
for root, dirs, files in os.walk(dir, topdown=False):
   for name in files:
      if name.endswith(".log"):
        with open(os.path.join(root, name), "r") as f:
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
                        id = random.choice(villagers_id)
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
            role_list = []
            for i in range(num_player):
                role_list.append(ROLES_DICT[id_role[i]])
            role_list = torch.Tensor(role_list, dtype=torch.int)
            data.append((game_status, role_list))

print("{} games loaded.".format(len(data)))
print(data[0][0])
print(len(data[0]))
torch.save(data, "logdata.pt")