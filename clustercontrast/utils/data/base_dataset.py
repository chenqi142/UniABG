import numpy as np

class BaseDataset(object):
    """
    Base class of CVGL dataset
    """

    def get_inmagedata_info(self, data):
        ids = []
        for _, id in data:
            ids += [id]
        ids = set(ids)
        num_ids = len(ids)
        num_imgs = len(data)
        return num_ids, num_imgs

    def print_dataset_statistics(self):
        raise NotImplementedError

    @property
    def images_dir(self):
        return None

class BaseImageDataset(BaseDataset):
    """
    Base class of image CVGL dataset
    """

    def print_dataset_statistics(self, train, logger=None):
        num_train_ids, num_train_imgs = self.get_inmagedata_info(train)


        logger.info("Dataset statistics:")
        logger.info("  -----------------------------")
        logger.info("  subset   | # ids | # images")
        logger.info("  -----------------------------")
        logger.info("  train    | {:5d} | {:8d}".format(num_train_ids, num_train_imgs))
        logger.info("  ----------------------------------------")