# OpenCV

## OpenCV 2

installed with python 2

## OpenCV 3

installed with python 3, but the `imshow` doesn't work, solve by 

```
conda install -c menpo opencv3
```

refer to [OpenCV not working properly with python on Linux with anaconda. Getting error that cv2.imshow() is not implemented
Ask Question](https://stackoverflow.com/questions/40207011/opencv-not-working-properly-with-python-on-linux-with-anaconda-getting-error-th)

## OpenCV 4

build from source according to the [official documentation](https://docs.opencv.org/4.0.1/d2/de6/tutorial_py_setup_in_ubuntu.html)

curious how the python distinguish different versions of opencv, does it affect the conda environment?

build from source by specifying something, refer to [Buidling OpenCV with Conda on Linux](http://alexsm.com/building-opencv-with-conda/) and [How to install OpenCV 4 on Ubuntu](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/)

1. 这并不意味着只安装在具体 conda envs 中，而是 /usr/local 中，最后再 ln -s 过去
2. `ln -s` 时python site-packages 下的 .so 文件必须重命名为 cv2.so，否则找不到！！
3. 所有其实一开始指定 cmake，只是为了针对具体版本进行编译，最后还需要自己 link，参照 [Compile OpenCV with Cmake to integrate it within a Conda env](https://stackoverflow.com/questions/50816241/compile-opencv-with-cmake-to-integrate-it-within-a-conda-env) 设置 `BUILD_opencv_python2=OFF`
