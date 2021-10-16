# Scripts

## Perspective Transform

For convenience, define the script as a function in `.bashrc`

```bash
persp_transform () 
{ 
    python ~/github/techNotes/src/persp.py "$1"
}
```

to use it 

```bash
$ conda activate py38
$ persp_transform folder "IMG_*.jpg"
```