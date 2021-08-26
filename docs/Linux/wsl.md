# Windows Subsytem Linux Ubuntu

## Mount USB drive

By default, the usb drive is not mounted to WSL, although it can be easily accessed via Windows.

```bash
# step 1: create a folder for mount
$ sudo mkdir /mnt/f
# step 2: mount
$ sudo mount -t drvfs F: /mnt/f
```

refer to [How to access my usb drive in Windows Subsytem Linux Ubuntu](https://askubuntu.com/questions/1116200/how-to-access-my-usb-drive-in-windows-subsytem-linux-ubuntu)

But when using `rsync` for copying files, it throws warnings that cannot set times on the folder and cannot `mkstemp`, although `scp` works fine. The natural solution is to change the permission, but `chown` not work. Then following [Rsync operation not permitted](https://unix.stackexchange.com/questions/558235/rsync-operation-not-permitted), remount the driver by specifying the user id and group id,

```bash
$ sudo unmount 
$ sudo mount -t drvfs -o rw,noatime,uid=1000,gid=1000 'P:\' /mnt/p
```

It is reported that [`scp` is slower than `rsync`](https://unix.stackexchange.com/questions/238152/why-is-scp-so-slow-and-how-to-make-it-faster), but based on my tries, it seems `scp` would be faster. But a big warning is that

!!! warning
    DO NOT interrupt a large file copy process, otherwise your PC would break down, and cannot ssh into. The only way is to restart manually. 
