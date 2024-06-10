py_library(
    name = "vid2avatar_lib",
    srcs = glob(["v2a_model.py", "lib/**/*.py"]),
    visibility = ["//visibility:public"],
    deps = [
        "//frames_folder/helpers:helpers_lib",    
        "//reconstruction/helpers:helpers_lib",
        "//reconstruction/datasets:datasets_lib",
        "//reconstruction/neuralbody:neuralbody_lib",
        "//reconstruction/smpl_fitting:smpl_fitting_lib",
    ],
    data = glob(["lib/smpl/smpl_model/**/*.pkl", "lib/model/**/*.pth"]),
)

py_binary(
    name = "vid2avatar_train",
    main = "train.py",
    srcs = ["train.py"],
    deps = ["//reconstruction/vid2avatar:vid2avatar_lib"],
)

py_binary(
    name = "vid2avatar_ff_artifacts_extract",
    main = "ff_extract_artifacts.py",
    srcs = ["ff_extract_artifacts.py"],
    deps = ["//reconstruction/vid2avatar:vid2avatar_lib"],
)

py_binary(
    name = "vid2avatar_test",
    main = "test.py",
    srcs = ["test.py"],
    deps = ["//reconstruction/vid2avatar:vid2avatar_lib"],
)
