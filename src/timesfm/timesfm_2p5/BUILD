# BUILD file for TimesFM 2.5 model implementations.

package(
    default_visibility = ["//learning/timeseries/scalar/timesfm:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "timesfm_2p5",
    srcs = ["__init__.py"],
    deps = [
        ":timesfm_2p5_base",
        ":timesfm_2p5_flax",
        ":timesfm_2p5_keras",
        ":timesfm_2p5_torch",
    ],
)

py_library(
    name = "timesfm_2p5_base",
    srcs = ["timesfm_2p5_base.py"],
    deps = [
        "//learning/timeseries/scalar/timesfm/tmax:configs",
    ],
)

py_library(
    name = "timesfm_2p5_flax",
    srcs = ["timesfm_2p5_flax.py"],
    deps = [
        ":timesfm_2p5_base",
        "//learning/timeseries/scalar/timesfm/tmax:configs",
        "//learning/timeseries/scalar/timesfm/tmax/flax:dense",
        "//learning/timeseries/scalar/timesfm/tmax/flax:transformer",
        "//learning/timeseries/scalar/timesfm/tmax/flax:util",
        "@org_einshape",
        "@org_jax//jax",
        "@org_jaxtyping//:jaxtyping",
        "@org_typeguard//:typeguard",
    ],
)

py_library(
    name = "timesfm_2p5_torch",
    srcs = ["timesfm_2p5_torch.py"],
    deps = [
        ":timesfm_2p5_base",
        "//learning/timeseries/scalar/timesfm/tmax:configs",
        "//learning/timeseries/scalar/timesfm/tmax/torch:dense",
        "//learning/timeseries/scalar/timesfm/tmax/torch:transformer",
        "//learning/timeseries/scalar/timesfm/tmax/torch:util",
        "@org_numpy//:numpy",
        "@org_pytorch//:torch",
        "@org_safetensors//:safetensors",
    ],
)

py_library(
    name = "timesfm_2p5_keras",
    srcs = ["timesfm_2p5_keras.py"],
    deps = [
        ":timesfm_2p5_base",
        "//learning/timeseries/scalar/timesfm/tmax:configs",
        "//learning/timeseries/scalar/timesfm/tmax/keras:dense",
        "//learning/timeseries/scalar/timesfm/tmax/keras:transformer",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
