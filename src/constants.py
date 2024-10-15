# Global constants

# Dataset locations
IMDB_SVCD = "~/dataset/SVCD/"
IMDB_LEVIRCD = "~/dataset/LEVIR-CD/256x256_in_1024x1024/"
IMDB_WHU = "~/dataset/WHU/256x256_random_test_code/"
IMDB_GZCD = "~/dataset/GZCD_256/GZCD_256_random"    #   本来分割成3603张图像的，最后三张去掉了，留下3600

# Template strings
CKP_LATEST = "checkpoint_latest.pth"
CKP_BEST = "model_best.pth"
CKP_COUNTED = "checkpoint_{e:03d}.pth"
CKP_BEST_THREE = "number{bestnum:1d}best_model.pth"