def map_labels_to_global(data, num_tasks):
    for task_ind in range(1, num_tasks + 1):
        global_to_local_map = data["metadata"]["global_to_local_class_mappings"][f"task_{task_ind}"]
        local_to_global_map = {v: int(k) for k, v in global_to_local_map.items()}

        for mode in ["train", "test"]:
            data[f"task_{task_ind}_{mode}"] = data[f"task_{task_ind}_{mode}"].map(
                lambda row: {"y": local_to_global_map[row["y"].item()]},
                desc="Mapping labels back to global.",
            )


def get_shared_samples_ids(data, num_tasks, shared_classes):
    """
    Get shared samples indices
    """

    # Add *shared* column, True for samples belonging to shared classes and False otherwise
    for task_ind in range(num_tasks + 1):
        for mode in ["train", "test"]:
            data[f"task_{task_ind}_{mode}"] = data[f"task_{task_ind}_{mode}"].map(
                lambda row: {"shared": row["y"].item() in shared_classes},
                desc="Adding shared column to samples",
            )

    shared_ids = []

    for task_ind in range(num_tasks + 1):
        all_ids = data[f"task_{task_ind}_train"]["id"]

        # get the indices of samples having shared to True
        task_shared_ids = all_ids[data[f"task_{task_ind}_train"]["shared"]].tolist()

        shared_ids.append(sorted(task_shared_ids))

    check_same_shared_ids(num_tasks, shared_ids)

    shared_ids = shared_ids[0]

    return shared_ids


def check_same_shared_ids(num_tasks, shared_ids):
    """
    Verify that each task has the same shared IDs
    """
    for task_i in range(num_tasks + 1):
        for task_j in range(task_i, num_tasks + 1):
            assert shared_ids[task_i] == shared_ids[task_j]
