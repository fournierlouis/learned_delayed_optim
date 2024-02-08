import gin
import jax
from typing import Tuple
from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.datasets import base
from learned_optimization.tasks.fixed.conv import _ConvTask, _cross_entropy_pool_loss
from learned_optimization.tasks.fixed.image_mlp import _MLPImageTask
from learned_optimization.tasks.fixed.transformer_lm import _TransformerTask
from learned_optimization.tasks.fixed.vit import (VisionTransformerTask, wide16_config,
            tall16_config) #vit_p16_h128_m512_nh4_nl10_config, deit_tiny_config, deit_small_config)
from learned_optimization.tasks.fixed.vit_test import VITTest
from learned_optimization.tasks.parametric.image_resnet import ParametricImageResNet
from learned_optimization.tasks.resnet import ResNet
from learned_optimization.tasks.fixed.resnet import _ResnetTask#Dataset



@base.dataset_lru_cache
@gin.configurable
def imagenet_datasets(
    batch_size: int,
    image_size: Tuple[int, int] = (224, 224),
    **kwargs,
) -> base.Datasets:
    splits = ("train", "validation", "validation", "test")
    return base.tfds_image_classification_datasets(
        datasetname="imagenet2012",
        splits=splits,
        batch_size=batch_size,
        image_size=image_size,
        stack_channels=1,
        prefetch_batches=50,
        shuffle_buffer_size=10000,
        normalize_mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
        normalize_std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
        convert_to_black_and_white=False,
        **kwargs,
    )


@base.dataset_lru_cache
@gin.configurable
def imagenet_64_datasets(
    batch_size: int,
    image_size: Tuple[int, int] = (64, 64),
    prefetch_batches=50,
    data_fraction=1.0,
    **kwargs,
) -> base.Datasets:
    perc = max(1, int(80 * data_fraction))
    splits = (f"train[0:{perc}%]", "train[80%:90%]", "train[90%:]", "validation")
    return base.tfds_image_classification_datasets(
        datasetname="imagenet_resized",
        splits=splits,
        batch_size=batch_size,
        image_size=image_size,
        stack_channels=1,
        prefetch_batches=prefetch_batches,
        shuffle_buffer_size=10000,
        normalize_mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
        normalize_std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
        convert_to_black_and_white=False,
        # cache=True,
        **kwargs,
    )


@gin.configurable
def My_Conv_Food101_32x64x64(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    base_model_fn = _cross_entropy_pool_loss([32, 64, 64], jax.nn.relu, num_classes=101)
    datasets = image.food101_datasets(batch_size=batch_size)
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def My_Conv_Cifar10_32x64x64(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    base_model_fn = _cross_entropy_pool_loss([32, 64, 64], jax.nn.relu, num_classes=10)
    datasets = image.cifar10_datasets(batch_size=batch_size, prefetch_batches=5)
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def My_Conv_Cifar10_8_16x32(batch_size):
    """A 2 hidden layer convnet designed for 8x8 cifar10."""
    base_model_fn = _cross_entropy_pool_loss([16, 32], jax.nn.relu, num_classes=10)
    datasets = image.cifar10_datasets(
        batch_size=batch_size,
        image_size=(8, 8),
    )
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def My_ImageMLP_Imagenet_Relu128x128(batch_size):
    """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(32, 32),
    )
    return _MLPImageTask(datasets, [128, 128])


@gin.configurable
def My_ImageMLP_Cifar10_Relu128x128(batch_size):
    """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
    datasets = image.cifar10_datasets(
        batch_size=batch_size,
        prefetch_batches=50,
    )
    return _MLPImageTask(datasets, [128, 128])


@gin.configurable
def My_ImageMLP_FashionMnist_Relu128x128(batch_size):
    """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
    datasets = image.fashion_mnist_datasets(batch_size=batch_size, prefetch_batches=5)
    return _MLPImageTask(datasets, [128, 128])


@gin.configurable
def My_ImageMLP_FashionMnist_Relu64x64(batch_size):
    """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
    datasets = image.fashion_mnist_datasets(batch_size=batch_size)
    return _MLPImageTask(datasets, [64, 64])


@gin.configurable
def My_ImageMLP_FashionMnist_Relu32x32(batch_size):
    """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
    datasets = image.fashion_mnist_datasets(batch_size=batch_size)
    return _MLPImageTask(datasets, [32, 32])


@gin.configurable
def My_Conv_FashionMnist_28_16x32(batch_size):
    """A 2 hidden layer, 128 hidden unit MLP designed for 28x28 fashion mnist."""
    datasets = image.fashion_mnist_datasets(batch_size=batch_size)
    base_model_fn = _cross_entropy_pool_loss([16, 32], jax.nn.relu, num_classes=10)
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def My_ImageMLP_FashionMnist8_Relu32(batch_size):
    """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 fashion mnist."""
    datasets = image.fashion_mnist_datasets(
        batch_size=batch_size,
        image_size=(8, 8),
    )
    return _MLPImageTask(datasets, [32])


###
# fmst
###


@gin.configurable
def conv_fmnist_32(batch_size):
    base_model_fn = _cross_entropy_pool_loss([32, 64, 64], jax.nn.relu, num_classes=10)
    datasets = image.fashion_mnist_datasets(batch_size=batch_size)
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def mlp128x128_fmnist_32(batch_size):
    datasets = image.fashion_mnist_datasets(batch_size=batch_size,prefetch_batches=1000)
    return _MLPImageTask(datasets, [128, 128])

@gin.configurable
def mlp512x512_fmnist_32(batch_size):
    datasets = image.fashion_mnist_datasets(batch_size=batch_size)
    return _MLPImageTask(datasets, [512, 512])

def mlp128_pow6_fmnist_32(batch_size):
    datasets = image.fashion_mnist_datasets(batch_size=batch_size)
    return _MLPImageTask(datasets, [128, 128, 128, 128, 128, 128])

@gin.configurable
def mlp64x64_fmnist_32(batch_size):
    datasets = image.fashion_mnist_datasets(batch_size=batch_size)
    return _MLPImageTask(datasets, [64, 64])


@gin.configurable
def mlp32x32_fmnist_32(batch_size):
    datasets = image.fashion_mnist_datasets(batch_size=batch_size)
    return _MLPImageTask(datasets, [32, 32])


@gin.configurable
def mlp128x128x128_fmnist_32(batch_size):
    datasets = idatasets = image.fashion_mnist_datasets(batch_size=batch_size)
    return _MLPImageTask(datasets, [128, 128, 128])


@gin.configurable
def small_conv_fmnist_8(batch_size):
    base_model_fn = _cross_entropy_pool_loss([16, 32], jax.nn.relu, num_classes=10)
    datasets = datasets = image.fashion_mnist_datasets(
        batch_size=batch_size,
        image_size=(8, 8),
    )
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def mlp32_fmnist_8(batch_size):
    datasets = datasets = image.fashion_mnist_datasets(
        batch_size=batch_size,
        image_size=(8, 8),
    )
    return _MLPImageTask(datasets, [32])


@gin.configurable
def mlp32x32_fmnist_8(batch_size):
    datasets = datasets = image.fashion_mnist_datasets(
        batch_size=batch_size,
        image_size=(8, 8),
    )
    return _MLPImageTask(datasets, [32, 32])


###
# Cifar-10
###


@gin.configurable
def conv_c10_32(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    base_model_fn = _cross_entropy_pool_loss([32, 64, 64], jax.nn.relu, num_classes=10)
    datasets = image.cifar10_datasets(
        batch_size=batch_size,
        prefetch_batches=50,
    )
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def mlp128x128_c10_32(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    datasets = image.cifar10_datasets(
        batch_size=batch_size,
        prefetch_batches=500,
    )
    return _MLPImageTask(datasets, [128, 128])


@gin.configurable
def mlp64x64_c10_32(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    datasets = image.cifar10_datasets(
        batch_size=batch_size,
        prefetch_batches=500,
    )
    return _MLPImageTask(datasets, [64, 64])


@gin.configurable
def mlp32x32_c10_32(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    datasets = image.cifar10_datasets(
        batch_size=batch_size,
        prefetch_batches=500,
    )
    return _MLPImageTask(datasets, [32, 32])


@gin.configurable
def mlp128x128x128_c10_32(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    datasets = image.cifar10_datasets(
        batch_size=batch_size,
        prefetch_batches=100,
    )
    return _MLPImageTask(datasets, [128, 128, 128])


@gin.configurable
def small_conv_c10_8(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    base_model_fn = _cross_entropy_pool_loss([16, 32], jax.nn.relu, num_classes=10)
    datasets = image.cifar10_datasets(
        batch_size=batch_size,
        prefetch_batches=50,
        image_size=(8, 8),
    )
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def mlp32_c10_8(batch_size):
    datasets = image.cifar10_datasets(
        batch_size=batch_size,
        prefetch_batches=50,
        image_size=(8, 8),
    )
    return _MLPImageTask(datasets, [32])


@gin.configurable
def mlp32x32_c10_8(batch_size):
    datasets = image.cifar10_datasets(
        batch_size=batch_size,
        prefetch_batches=50,
        image_size=(8, 8),
    )
    return _MLPImageTask(datasets, [32, 32])


###
# Imagenet
###


@gin.configurable
def mlp128x128_imagenet_32(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(32, 32),
    )
    return _MLPImageTask(datasets, [128, 128])

@gin.configurable
def mlp16x16x16_imagenet_32(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(32, 32),
    )
    return _MLPImageTask(datasets, [16, 16, 16])

@gin.configurable
def mlp64x64x64_imagenet_32(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(32, 32),
    )
    return _MLPImageTask(datasets, [64, 64, 64])

@gin.configurable
def mlp256x256x256_imagenet_32(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(32, 32),
    )
    return _MLPImageTask(datasets, [256, 256, 256])

@gin.configurable
def mlp1024x1024x1024_imagenet_32(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(32, 32),
    )
    return _MLPImageTask(datasets, [1024, 1024, 1024])

@gin.configurable
def mlp4096x4096x4096_imagenet_32(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(32, 32),
    )
    return _MLPImageTask(datasets, [4096, 4096, 4096])

@gin.configurable
def mlp128x128_imagenet_64(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    datasets = imagenet_64_datasets(
        batch_size=batch_size, image_size=(64, 64), prefetch_batches=5
    )
    return _MLPImageTask(datasets, [128, 128])


@gin.configurable
def mlp64x64_imagenet_64(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(64, 64),
    )
    return _MLPImageTask(datasets, [64, 64])

@gin.configurable
def conv32x32_imagenet_64(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    base_model_fn = _cross_entropy_pool_loss([32, 32], jax.nn.relu, num_classes=10)
    datasets = imagenet_64_datasets(
        batch_size=batch_size, image_size=(64, 64), prefetch_batches=200
    )
    return _ConvTask(base_model_fn, datasets)

@gin.configurable
def conv_imagenet_32(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    base_model_fn = _cross_entropy_pool_loss(
        [32, 64, 64], jax.nn.relu, num_classes=1000
    )
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(32, 32),
    )
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def small_conv_imagenet_8(batch_size):
    base_model_fn = _cross_entropy_pool_loss([16, 32], jax.nn.relu, num_classes=1000)
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(8, 8),
    )
    return _ConvTask(base_model_fn, datasets)


@gin.configurable
def mlp32_imagenet_8(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(8, 8),
    )
    return _MLPImageTask(datasets, [32])


@gin.configurable
def mlp32x32_imagenet_8(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(8, 8),
    )
    return _MLPImageTask(datasets, [32, 32])


def tall16_imagenet_32(batch_size):
    model = tall16_config()
    datasets = imagenet_64_datasets(
        batch_size=batch_size, image_size=(32, 32), prefetch_batches=500
    )
    return VisionTransformerTask(model, datasets)



def vit_p16_h128_m512_nh4_nl10_imagenet_32(batch_size):
    model = vit_p16_h128_m512_nh4_nl10_config()
    datasets = imagenet_64_datasets(
        batch_size=batch_size, image_size=(32, 32), prefetch_batches=50
    )
    return VisionTransformerTask(model, datasets)


def tall16_imagenet_64(batch_size):
    model = tall16_config()
    datasets = imagenet_64_datasets(
        batch_size=batch_size, image_size=(64, 64), prefetch_batches=50
    )
    return VisionTransformerTask(model, datasets)

def tall16_imagenet_8(batch_size):
    model = tall16_config()
    datasets = imagenet_64_datasets(
        batch_size=batch_size, image_size=(8 ,8), prefetch_batches=500
    )
    return VisionTransformerTask(model, datasets)



def wide16_imagenet_32(batch_size):
    model = wide16_config()
    datasets = imagenet_64_datasets(
        batch_size=batch_size, image_size=(32, 32), prefetch_batches=500
    )
    return VisionTransformerTask(model, datasets)

def wide16_imagenet_8(batch_size):
    model = wide16_config()
    datasets = imagenet_64_datasets(
        batch_size=batch_size, image_size=(8 ,8), prefetch_batches=500
    )
    return VisionTransformerTask(model, datasets)


# def resnet18_imagenet_32(batch_size):
#     datasets = imagenet_64_datasets(
#         batch_size=batch_size, image_size=(32, 32), prefetch_batches=500
#     )
#     resnet18 = {
#           "blocks_per_group": (2, 2, 2, 2),
#         #   "bottleneck": False,
#           "initial_conv_channels":64,  
#           "initial_conv_stride":2,
#           "initial_conv_kernel_size":7,
#         #   "blocks_per_group":,
#         #   "channels_per_group":,
#           "max_pool":True,
#           "channels_per_group": (64, 128, 256, 512),
#         #   "use_projection": (False, True, True, True),
#       }

#     print(resnet18)
#     return ParametricImageResNet(datasets,**resnet18)



def resnet18_imagenet_32(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size, image_size=(32, 32), prefetch_batches=50
    )
    task = _ResnetTask(cfg=dict(batch_size=batch_size,image_size=32,
                                initial_conv_kernel_size=7,initial_conv_stride=2,resnet_v2=False, max_pool=True,
                                **ResNet.CONFIGS[18]))
    task.datsets = datasets
    return task

def resnet18_imagenet_64(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size, image_size=(64, 64), prefetch_batches=50
    )
    task = _ResnetTask(cfg=dict(batch_size=batch_size,image_size=64,
                                initial_conv_kernel_size=7,initial_conv_stride=2,resnet_v2=False, max_pool=True,
                                **ResNet.CONFIGS[18]))
    task.datsets = datasets
    return task


def resnet50_imagenet_32(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size, image_size=(32, 32), prefetch_batches=50
    )
    task = _ResnetTask(cfg=dict(batch_size=batch_size,image_size=32,
                                initial_conv_kernel_size=7,initial_conv_stride=2,resnet_v2=False, max_pool=True,
                                **ResNet.CONFIGS[50]))
    task.datsets = datasets
    return task



def resnet50_imagenet_128(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size, image_size=(128, 128), prefetch_batches=20
    )
    task = _ResnetTask(cfg=dict(batch_size=batch_size,image_size=128,
                                initial_conv_kernel_size=7,initial_conv_stride=2,resnet_v2=False, max_pool=True,
                                **ResNet.CONFIGS[50]))
    task.datsets = datasets
    return task



def resnet50_imagenet_64(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size, image_size=(64, 64), prefetch_batches=50
    )
    task = _ResnetTask(cfg=dict(batch_size=batch_size,image_size=64,
                                initial_conv_kernel_size=7,initial_conv_stride=2,resnet_v2=False, max_pool=True,
                                **ResNet.CONFIGS[50]))
    task.datsets = datasets
    return task





def deit_tiny_imagenet_64(batch_size):
    model = deit_tiny_config()
    datasets = imagenet_64_datasets(
        batch_size=batch_size, image_size=(64, 64), prefetch_batches=50
    )
    return VisionTransformerTask(model, datasets)


def deit_small_imagenet_64(batch_size):
    model = deit_small_config()
    datasets = imagenet_64_datasets(
        batch_size=batch_size, image_size=(64, 64), prefetch_batches=50
    )
    return VisionTransformerTask(model, datasets)






# study generalization to width, depth, and larger images 

@gin.configurable
def mlp128_pow6_imagenet_32(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(32, 32),
    )
    return _MLPImageTask(datasets, [128, 128, 128, 128, 128, 128])


@gin.configurable
def mlp128_pow12_imagenet_32(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(32, 32),
    )
    return _MLPImageTask(datasets, [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128])


@gin.configurable
def mlp128_imagenet_32(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(32, 32),
    )
    return _MLPImageTask(datasets, [128])

@gin.configurable
def mlp128x128x128_imagenet_32(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(32, 32),
    )
    return _MLPImageTask(datasets, [128, 128, 128])


@gin.configurable
def mlp512x512x512_imagenet_32(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(32, 32),
    )
    return _MLPImageTask(datasets, [512, 512, 512])



@gin.configurable
def mlp128x128x128_imagenet_64(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(64, 64),
    )
    return _MLPImageTask(datasets, [128, 128, 128])


@gin.configurable
def mlp128x128x128_imagenet_128(batch_size):
    datasets = imagenet_64_datasets(
        batch_size=batch_size,
        image_size=(128, 128),
    )
    return _MLPImageTask(datasets, [128, 128, 128])


@gin.configurable
def mlp128x128x128_c100_32(batch_size):
    """A 3 hidden layer convnet designed for 32x32 cifar10."""
    datasets = image.cifar100_datasets(
        batch_size=batch_size,
        #    prefetch_batches=10,
    )
    return _MLPImageTask(datasets, [128, 128, 128])


# LM
def transformer32_lm(batch_size):
    _d_model = 32
    _cfg = {
        "num_heads": 4,
        "d_model": _d_model,
        "num_layers": 2,
        "batch_size": batch_size,
        "sequence_length": 8,
        "dropout_rate": 0.1,
    }
    _task_name = "TransformerLM_LM1B_5layer_%dwidth" % _d_model
    return _TransformerTask(_cfg, name=_task_name)



def transformer192_lm(batch_size):
    _d_model = 192
    _cfg = {
        "num_heads": 12,
        "d_model": _d_model,
        "num_layers": 12,
        "batch_size": batch_size,
        "sequence_length": 16,
        "dropout_rate": 0.1,
    }
    _task_name = "TransformerLM_LM1B_5layer_%dwidth" % _d_model
    return _TransformerTask(_cfg, name=_task_name)


def get_task(args, is_test=False):
    tasks = {
        'transformer192_lm':transformer192_lm,
        'resnet18_imagenet_64':resnet18_imagenet_64,
        'resnet18_imagenet_32':resnet18_imagenet_32,
        'resnet50_imagenet_32':resnet50_imagenet_32,


        'resnet50_imagenet_128':resnet50_imagenet_128,
        'resnet50_imagenet_64':resnet50_imagenet_64,
        'deit_tiny_imagenet_64':deit_tiny_imagenet_64,
        'deit_small_imagenet_64':deit_small_imagenet_64,


        'mlp512x512_fmnist_32':mlp512x512_fmnist_32,
        'mlp128_pow6_fmnist_32':mlp128_pow6_fmnist_32,
        "mlp128_pow12_imagenet_32": mlp128_pow12_imagenet_32,
        "mlp128x128x128_imagenet_128":mlp128x128x128_imagenet_128,
        "mlp128x128x128_imagenet_64":mlp128x128x128_imagenet_64,
        "mlp512x512x512_imagenet_32":mlp512x512x512_imagenet_32,
        'mlp256x256x256_imagenet_32':mlp256x256x256_imagenet_32,
        "mlp64x64x64_imagenet_32":mlp64x64x64_imagenet_32,
        "mlp128_imagenet_32":mlp128_imagenet_32,
        "mlp128_pow6_imagenet_32":mlp128_pow6_imagenet_32,
        "mlp128x128x128_c100_32": mlp128x128x128_c100_32,
        # "VIT_Cifar100_wideshallow": VITTest.test_tasks("VIT_Cifar100_wideshallow"),
        "vit_p16_h128_m512_nh4_nl10_imagenet_32":vit_p16_h128_m512_nh4_nl10_imagenet_32,
        "deit_tiny_imagenet_64":deit_tiny_imagenet_64,
        # "deit_tiny_imagenet_224":deit_tiny_imagenet_224,
        "tall16_imagenet_64":tall16_imagenet_64,
        "tall16_imagenet_32": tall16_imagenet_32,
        "tall16_imagenet_8": tall16_imagenet_8,
        "wide16_imagenet_32": wide16_imagenet_32,
        "wide16_imagenet_8": wide16_imagenet_8,
        # LM
        "transformer32_lm": transformer32_lm,
        # 8x8 fmst
        "small-image-mlp-fmst": My_ImageMLP_FashionMnist8_Relu32,
        "conv-fmst": My_Conv_Cifar10_32x64x64,
        "image-mlp-fmst": My_ImageMLP_FashionMnist_Relu128x128,
        "image-mlp-fmst64x64": My_ImageMLP_FashionMnist_Relu64x64,
        "image-mlp-fmst32x32": My_ImageMLP_FashionMnist_Relu32x32,
        # cifar10
        "image-mlp-c10-128x128": My_ImageMLP_Cifar10_Relu128x128,
        "conv-c10": My_Conv_Cifar10_32x64x64,
        "small-conv-c10": My_Conv_Cifar10_8_16x32,
        # inet
        "conv32x32_imagenet_64": conv32x32_imagenet_64,
        "mlp128x128_imagenet_64": mlp128x128_imagenet_64,
        "mlp64x64_imagenet_64": mlp64x64_imagenet_64,
        "conv_imagenet_32": conv_imagenet_32,
        "mlp128x128_imagenet_32": mlp128x128_imagenet_32,
        "mlp128x128x128_imagenet_32": mlp128x128x128_imagenet_32,
        "small_conv_imagenet_8": small_conv_imagenet_8,
        "mlp32_imagenet_8": mlp32_imagenet_8,
        "mlp32x32_imagenet_8": mlp32x32_imagenet_8,
        # c10
        "conv_c10_32": conv_c10_32,
        "mlp128x128_c10_32": mlp128x128_c10_32,
        "mlp64x64_c10_32": mlp64x64_c10_32,
        "mlp32x32_c10_32": mlp32x32_c10_32,
        "mlp128x128x128_c10_32": mlp128x128x128_c10_32,
        "small_conv_c10_8": small_conv_c10_8,
        "mlp32_c10_8": mlp32_c10_8,
        "mlp32x32_c10_8": mlp32x32_c10_8,
        # fmst
        "conv_fmnist_32": conv_fmnist_32,
        "mlp128x128_fmnist_32": mlp128x128_fmnist_32,
        "mlp64x64_fmnist_32": mlp64x64_fmnist_32,
        "mlp32x32_fmnist_32": mlp32x32_fmnist_32,
        "mlp128x128x128_fmnist_32": mlp128x128x128_fmnist_32,
        "small_conv_fmnist_8": small_conv_fmnist_8,
        "mlp32_fmnist_8": mlp32_fmnist_8,
        "mlp32x32_fmnist_8": mlp32x32_fmnist_8,
        # multiple tasks
        "fmnist-conv-mlp-mix": [
            My_Conv_FashionMnist_28_16x32,
            My_ImageMLP_FashionMnist_Relu64x64,
            My_ImageMLP_FashionMnist_Relu128x128,
        ],
        "fmnist-mlp-mix": [
            mlp128x128_fmnist_32,
            mlp64x64_fmnist_32,
            mlp32x32_fmnist_32,
        ],
        "dataset-mlp-mix": [
            My_ImageMLP_Imagenet_Relu128x128,
            My_ImageMLP_Cifar10_Relu128x128,
            My_ImageMLP_FashionMnist_Relu128x128,
        ],
        # cifar-fmnist-all
        "cifar-fmnist-all": [
            mlp128x128_fmnist_32,
            mlp64x64_fmnist_32,
            mlp32x32_fmnist_32,
            mlp128x128_c10_32,
            mlp64x64_c10_32,
            mlp32x32_c10_32,
        ],
        "cifar-fmnist-128x128": [
            mlp128x128_fmnist_32,
            mlp128x128_c10_32,
        ],
    }

    test_batch_size = {
        "small-conv-imagenet32": 10000,
        "conv-imagenet32": 10000,
        "small-conv-imagenet8": 10000,
        "conv-imagenet8": 10000,
        "image-mlp-imagenet32-128x128": 10000,
        "image-mlp-c10-128x128": 10000,
        "image-mlp-fmst": 10000,
        "image-mlp-fmst64x64": 10000,
        "image-mlp-fmst32x32": 10000,
        "small-image-mlp-fmst": 10000,
        "conv-c10": 10000,
        "small-conv-c10": 10000,
        "conv-imagenet64": 100000,  # TODO Could probably get oom error, fix it when needed
        "conv-imagenet": 100000,
        "fmnist-conv-mlp-mix": 10000,
        "fmnist-mlp-mix": 10000,
        "dataset-mlp-mix": 10000,
        "cifar-fmnist-128x128": 10000,
        "cifar-fmnist-all": 10000,
        "transformer32_lm": 10000,
    }
    test_batch_size.update(
        {
            k: 10000
            for k in tasks.keys()
            if ("_c10" in k or "imagenet" in k or "_fmnist" in k)
        }
    )
    batch_size = args.num_grads * args.num_local_steps * args.local_batch_size
    if is_test:
        batch_size = test_batch_size[args.task]

    task = tasks[args.task]

    if type(task) is list:
        return [task(batch_size) for task in task]
    else:
        return tasks[args.task](batch_size)
