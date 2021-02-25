import pickle
import random

file_path = "/home/nfs/andreaalfieria/thesis/detr/real_gates_lists/all_daylight_all_iros.pkl"

with open(file_path, 'rb') as file:
    file_list: list = pickle.load(file)

print(f"There are {len(file_list)} files in the list")

random.shuffle(file_list)
print("List shuffled")
with open(file_path, 'wb') as file:
    pickle.dump(file_list, file)
print("List updated")