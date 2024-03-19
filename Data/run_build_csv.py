from BuildCSVFromDir import BuildCSVFromDir
import hydra
from omegaconf import DictConfig

@hydra.main(config_path='hydra_config', config_name='config')
def main(cfg: DictConfig):
    """
    Main function to build the csv file from the directory of images and masks.
    
    Args:
        cfg: Configuration for the dataset.
    """
    train_images = cfg.paths.base_path + '/' + cfg.paths.train_dir + '/' + cfg.paths.images_dir
    train_masks = cfg.paths.base_path + '/' + cfg.paths.train_dir + '/' + cfg.paths.masks_dir
    train_csv = cfg.paths.base_path + '/' + cfg.paths.train_dir + '/' + cfg.paths.train_csv

    test_images = cfg.paths.base_path + '/' + cfg.paths.test_dir + '/' + cfg.paths.images_dir
    test_masks = cfg.paths.base_path + '/' + cfg.paths.test_dir + '/' + cfg.paths.masks_dir
    test_csv = cfg.paths.base_path + '/' + cfg.paths.test_dir + '/' + cfg.paths.test_csv

    val_images = cfg.paths.base_path + '/' + cfg.paths.val_dir + '/' + cfg.paths.images_dir
    val_masks = cfg.paths.base_path + '/' + cfg.paths.val_dir + '/' + cfg.paths.masks_dir
    val_csv = cfg.paths.base_path + '/' + cfg.paths.val_dir + '/' + cfg.paths.val_csv

    build_csv_train = BuildCSVFromDir(train_images, train_masks, train_csv, mode='train')
    build_csv_train()

    build_csv_test = BuildCSVFromDir(test_images, test_masks, test_csv, mode='test')
    build_csv_test()

    build_csv_val = BuildCSVFromDir(val_images, val_masks, val_csv, mode='val')
    build_csv_val()

if __name__ == '__main__':
    main()