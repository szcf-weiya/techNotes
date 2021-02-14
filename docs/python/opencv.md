# OpenCV

In short,

```bash
$ pip install opencv-python
```

!!! info
    For example, the [workflow of Cell-Video](https://github.com/szcf-weiya/Cell-Video/blob/master/.github/workflows/blank.yml#L70) uses `requirements.py`, which includes `opencv-python`.

## OpenCV 2

installed with python 2

## OpenCV 3

installed with python 3, but the `imshow` doesn't work, solve by

```
conda install -c menpo opencv3
```

refer to [OpenCV not working properly with python on Linux with anaconda. Getting error that cv2.imshow() is not implemented
Ask Question](https://stackoverflow.com/questions/40207011/opencv-not-working-properly-with-python-on-linux-with-anaconda-getting-error-th)

If necessary, install [imutils: A series of convenience functions to make basic image processing operations such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images easier with OpenCV and Python.](https://github.com/jrosebr1/imutils)

```bash
conda install -c conda-forge imutils
```

Currently I am mainly work with

```bash
(py37) $ conda list
opencv                    3.4.2            py37h6fd60c2_1    defaults
opencv-python             4.2.0.34                 pypi_0    pypi
```

which can be easily installed with

```python
$ pip install opencv-python
```

pay attention to that `import cv2; cv2.__version__` returns the version of `opencv-python`. The `contrib` packages can be installed with (see, [https://pypi.org/project/opencv-contrib-python/](https://pypi.org/project/opencv-contrib-python/))

```bash
(py37) $ pip install opencv-contrib-python
```

### Some bugs

```python
error: OpenCV(3.4.2) /tmp/build/80754af9/opencv-suite_1535558553474/work/modules/imgproc/src/color.hpp:253: error: (-215:Assertion failed) VScn::contains(scn) && VDcn::contains(dcn) && VDepth::contains(depth) in function 'CvtHelper'
```

the image might be not read properly, such as wrong path.

## OpenCV 4

build from source according to the [official documentation](https://docs.opencv.org/4.0.1/d2/de6/tutorial_py_setup_in_ubuntu.html)

curious how the python distinguish different versions of opencv, does it affect the conda environment?

build from source by specifying something, refer to [Buidling OpenCV with Conda on Linux](http://alexsm.com/building-opencv-with-conda/) and [How to install OpenCV 4 on Ubuntu](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/)

1. 这并不意味着只安装在具体 conda envs 中，而是 /usr/local 中，最后再 ln -s 过去。换句话说，安装时使用的 python 环境仅用于 build。
2. `ln -s` 时python site-packages 下的 .so 文件必须重命名为 cv2.so，否则找不到！！
3. 所有其实一开始指定 cmake，只是为了针对具体版本进行编译，最后还需要自己 link，参照 [Compile OpenCV with Cmake to integrate it within a Conda env](https://stackoverflow.com/questions/50816241/compile-opencv-with-cmake-to-integrate-it-within-a-conda-env) 设置 `BUILD_opencv_python2=OFF`

Currently (2020-07-27 21:32:18), I am not using this version, and the previous built conda env also has been deleted due to the upgrade of system. But the build folders (in `~/github/opencv4_build`) and resulting `.so` (in `/usr/local/lib/python3.7/site-packages/cv2/python-3.7`) should still be OK since the beginning python is just for building.