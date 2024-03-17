from Preprocessing.ProcessingPolygons import ProcessingPolygons
import hydra
from omegaconf import DictConfig

@hydra.main(config_path='hydra_config', config_name='config.yaml', version_base=None)
def main(cfg: DictConfig):
    """
    Generate the entire dataset (train + val + test)
    """

    train_labels = cfg.paths.base_path + '/' + cfg.paths.train_dir + '/' + cfg.paths.labels_dir
    train_images = cfg.paths.base_path + '/' + cfg.paths.train_dir + '/' + cfg.paths.images_dir
    train_masks = cfg.paths.base_path + '/' + cfg.paths.train_dir + '/' + cfg.paths.masks_dir

    test_labels = cfg.paths.base_path + '/' + cfg.paths.test_dir + '/' + cfg.paths.labels_dir
    test_images = cfg.paths.base_path + '/' + cfg.paths.test_dir + '/' + cfg.paths.images_dir
    test_masks = cfg.paths.base_path + '/' + cfg.paths.test_dir + '/' + cfg.paths.masks_dir

    val_labels = cfg.paths.base_path + '/' + cfg.paths.val_dir + '/' + cfg.paths.labels_dir
    val_images = cfg.paths.base_path + '/' + cfg.paths.val_dir + '/' + cfg.paths.images_dir
    val_masks = cfg.paths.base_path + '/' + cfg.paths.val_dir + '/' + cfg.paths.masks_dir 

    preprocessing_polygons_train = ProcessingPolygons(train_labels, train_images, train_masks, mode='train')
    preprocessing_polygons_train()

    preprocessing_polygons_test = ProcessingPolygons(test_labels, test_images, test_masks, mode='test')
    preprocessing_polygons_test()

    preprocessing_polygons_val = ProcessingPolygons(val_labels, val_images, val_masks, mode='val')
    preprocessing_polygons_val()

if __name__ == '__main__':
    main()