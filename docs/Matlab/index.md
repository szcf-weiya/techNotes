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