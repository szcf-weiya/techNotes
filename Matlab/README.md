# ACF people detector

```
detector = peopleDetectorACF;
%detector = peopleDetectorACF('caltech');
I = imread('testACF.png');
[bboxes, scores] = detect(detector, I);
I = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
figure
imshow(I)
```

