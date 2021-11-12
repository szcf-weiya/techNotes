# Matlab相关

## ACF people detector

```
detector = peopleDetectorACF;
%detector = peopleDetectorACF('caltech');
I = imread('testACF.png');
[bboxes, scores] = detect(detector, I);
I = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
figure
imshow(I)
```

## Launch from Command Line

- `nodesktop`: still have full access to figure windows, and can launch the full desktop with the command desktop.
- `nojvm`: not only does not load the desktop, but does not load java, i.e., cannot later bring up the desktop, launch figure windows, or use any commands that use java libraries
- `nosplash`: skip the splash screen, the first window with the picture of the membrane and the version number. It still loads into the desktop.

refer to [Launching MATLAB without the desktop](https://blogs.mathworks.com/community/2010/02/22/launching-matlab-without-the-desktop/)

These are called [Startup Options](https://www.mathworks.com/help/matlab/matlab_env/startup-options.html), and the official Matlab describe the options for different systems, such as [matlab (Linux)](https://www.mathworks.com/help/matlab/ref/matlablinux.html), [matlab (Windows)](https://www.mathworks.com/help/matlab/ref/matlabwindows.html).

## Option `-wait`

I saw the option in the `.bat` file for Windows. By checking the documentation, [matlab (Windows)](https://www.mathworks.com/help/matlab/ref/matlabwindows.html) said,

> To capture the exit code, start MATLAB with the -wait option.

but no similar description in [matlab (Linux)](https://www.mathworks.com/help/matlab/ref/matlablinux.html) and [matlba (macOS)](https://www.mathworks.com/help/matlab/ref/matlabmacos.html).

## assign multiple variables

```matlab
[t1, t2, t3] = deal(1, 2, 3)
```

refer to [Assign Multiple Variables](https://www.mathworks.com/matlabcentral/answers/16996-assign-multiple-variables)