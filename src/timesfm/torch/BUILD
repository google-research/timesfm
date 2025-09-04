load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

pytype_strict_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":dense",
        ":transformer",
        ":util",
        "//learning/timeseries/scalar/timesfm/tmax:abstract",
        "//third_party/py/torch:pytorch",
    ],
)

pytype_strict_library(
    name = "transformer",
    srcs = ["transformer.py"],
    deps = [
        ":normalization",
        ":util",
        "//learning/timeseries/scalar/timesfm/tmax:abstract",
        "//third_party/py/torch:pytorch",
    ],
)

pytype_strict_library(
    name = "dense",
    srcs = ["dense.py"],
    deps = [
        "//learning/timeseries/scalar/timesfm/tmax:abstract",
        "//third_party/py/torch:pytorch",
    ],
)

pytype_strict_library(
    name = "util",
    srcs = ["util.py"],
    deps = ["//third_party/py/torch:pytorch"],
)

pytype_strict_library(
    name = "normalization",
    srcs = ["normalization.py"],
    deps = ["//third_party/py/torch:pytorch"],
)
