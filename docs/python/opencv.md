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

## OpenCV 4

build from source according to the [official documentation](https://docs.opencv.org/4.0.1/d2/de6/tutorial_py_setup_in_ubuntu.html)

curious how the python distinguish different versions of opencv, does it affect the conda environment?

build from source by specifying something, refer to [Buidling OpenCV with Conda on Linux](http://alexsm.com/building-opencv-with-conda/) and [How to install OpenCV 4 on Ubuntu](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/)

1. 这并不意味着只安装在具体 conda envs 中，而是 /usr/local 中，最后再 ln -s 过去。换句话说，安装时使用的 python 环境仅用于 build。
2. `ln -s` 时python site-packages 下的 .so 文件必须重命名为 cv2.so，否则找不到！！
3. 所有其实一开始指定 cmake，只是为了针对具体版本进行编译，最后还需要自己 link，参照 [Compile OpenCV with Cmake to integrate it within a Conda env](https://stackoverflow.com/questions/50816241/compile-opencv-with-cmake-to-integrate-it-within-a-conda-env) 设置 `BUILD_opencv_python2=OFF`

Currently (2020-07-27 21:32:18), I am not using this version, and the previous built conda env also has been deleted due to the upgrade of system. But the build folders (in `~/github/opencv4_build`) and resulting `.so` (in `/usr/local/lib/python3.7/site-packages/cv2/python-3.7`) should still be OK since the beginning python is just for building.

## Possible Solution for Errors

- check the image path carefully

```python
error: OpenCV(3.4.2) /tmp/build/80754af9/opencv-suite_1535558553474/work/modules/imgproc/src/color.hpp:253: error: (-215:Assertion failed) VScn::contains(scn) && VDcn::contains(dcn) && VDepth::contains(depth) in function 'CvtHelper'
```

the image might be not read properly, such as wrong path.

- create a completely new environment, and install `python-opencv`, such as [NULL window handler in function 'cvSetMouseCallback'](https://stackoverflow.com/questions/62801244/null-window-handler-in-function-cvsetmousecallback)

## PIL and Pillow

PIL is the Python Imaging Library, and [Pillow](https://pillow.readthedocs.io/en/stable/index.html) is the friendly PIL fork.
    - 由于PIL仅支持到Python 2.7，加上年久失修，于是一群志愿者在PIL的基础上创建了兼容的版本，名字叫Pillow，支持最新Python 3.x，又加入了许多新特性 [:link:](https://www.liaoxuefeng.com/wiki/1016959663602400/1017785454949568)

!!! info
    [An example](https://github.com/szcf-weiya/Clouds/issues/7).

## Read Image

!!! info
    Adopted from [my project](https://github.com/szcf-weiya/CTCalgs/issues/1).

For a 16-bit TIFF images, the [mode](https://docs.opencv.org/master/d6/d87/imgcodecs_8hpp.html) should be set as `IMREAD_ANYDEPTH` instead of `IMREAD_GRAYSCALE`.

```python
>>> im1 = cv2.imread("example/three_cells.tif", cv2.IMREAD_ANYDEPTH)
>>> np.unique(im1)
array([ 0, 46, 51, 52, 53, 54, 55, 56], dtype=uint16)
>>> im2 = cv2.imread("../test/example/three_cells.tif", cv2.IMREAD_GRAYSCALE)
>>> np.unique(im2)
array([0], dtype=uint8)
```

while `skimage.io` avoids such specification, which might be more convenience,

```python
>>> im3 = io.imread("../test/example/three_cells.tif")
>>> np.unique(im3)
array([ 0, 46, 51, 52, 53, 54, 55, 56], dtype=uint16)
```

As a comparison, JuliaImages uses a special type `NOf16` (standing for **N**ormalized, with **0** integer bits and **16 f**ractional bits), that interprets an 16-bit integer as if it had been scaled by 1/(2^16-1), thus encoding values from 0 to 1 in 2^16 steps.

```julia
julia> im = load("../test/example/three_cells.tif")
# in jupyter notebook, direct `unique(im)` will automatically shows the image pixel
julia> display(MIME("text/plain"), sort(unique(im)))
8-element Array{Gray{N0f16},1} with eltype Gray{Normed{UInt16,16}}:
 Gray{N0f16}(0.0)
 Gray{N0f16}(0.0007)
 Gray{N0f16}(0.00078)
 Gray{N0f16}(0.00079)
 Gray{N0f16}(0.00081)
 Gray{N0f16}(0.00082)
 Gray{N0f16}(0.00084)
 Gray{N0f16}(0.00085)
```

!!! warning
    The result returned by `np.unique` has been sorted, while Julia's `unique` does not.

As for Matlab, `imread` performs similarly as `io.imread`, returning the integer in `uint16` form,

```matlab
% original
>> oIm = imread("../Fluo-N2DH-SIM+/01/t000.tif")
>> [max(max(oIm)), min(min(oIm))]

ans =

  1×2 uint16 row vector

   314    73

% convert to double
>> doIm = double(oIm) / (2^16-1) * 255
>> [max(max(doIm)), min(min(doIm))]

ans =

    1.2218    0.2840

>> subplot(1, 2, 1), imshow(oIm)
>> subplot(1, 2, 2), imshow(doIm)
```

![](https://user-images.githubusercontent.com/13688320/99756556-664de800-2b28-11eb-902f-5d2b0ffc14d6.png)

use `cv2.imwrite()` for writting images from `np.array`, see also [Saving a Numpy array as an image](https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image/19174800)