import os
import os.path as osp
from ..utils.data.base_dataset import BaseImageDataset

def get_data(path):
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            folder_path = os.path.join(root, name)
            data[name] = {"path": folder_path, "files": os.listdir(folder_path)}
    return data

class university_drone(BaseImageDataset):
    dataset_dir = 'University-Release'

    def __init__(self, root, logger=None, verbose=True, **kwargs):
        super(university_drone, self).__init__()
        root = '/data0/chenqi_data'
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train/drone')
        self.drone_dir = get_data(self.train_dir)

        self.ids = list(set(self.drone_dir.keys()))
        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}

        self.dataset = []
        for idx in self.ids:
            for file in self.drone_dir[idx]["files"]:
                image_path = "{}/{}".format(self.drone_dir[idx]["path"], file)
                label = self.reverse_map_dict[idx]
                self.dataset.append((image_path, label))

        if verbose:
            logger.info("=> University-1652 drone loaded")
            self.print_dataset_statistics(self.dataset, logger=logger)
