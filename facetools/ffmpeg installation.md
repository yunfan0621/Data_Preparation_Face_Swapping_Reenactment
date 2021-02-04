# ffmpeg installation

## Ubuntu

```
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt-get update
sudo apt-get install ffmpeg
```

## Centos

`pip install ffmpeg-python`

### Step 1: Update the system

```
sudo yum install epel-release -y
sudo yum update -y
sudo shutdown -r now
```

### Step 2: Install the Nux Dextop YUM repo

There are no official FFmpeg rpm packages for CentOS for now. Instead, you can use a 3rd-party YUM repo, Nux Dextop, to finish the job.

On CentOS 7, you can install the Nux Dextop YUM repo with the following commands:

```
sudo rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
sudo rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm
```

For CentOS 6, you need to install another release:

```
sudo rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
sudo rpm -Uvh http://li.nux.ro/download/nux/dextop/el6/x86_64/nux-dextop-release-0-2.el6.nux.noarch.rpm
```

### Step 3: Install FFmpeg and FFmpeg development packages

```
sudo yum install ffmpeg ffmpeg-devel -y
```

### Step 4: Test drive

1) Confirm the installation of FFmpeg:

```
ffmpeg
```

This command provides detailed info about FFmpeg installed on your system. At the time of writing, the version of FFmpeg installed using Nux dextop is 2.6.8.

If you want to learn more about FFmpeg, input:

```
ffmpeg -h
```



```python
img1 = np.array(size=(512, 512, 3))
img2 = np.array(size=(512, 512, 3))

img1_tensor = torch.from_numpy(img1).permute(2,0,1).unsqueeze_(0) 
# img1_tensor shape = (1, 3, 512, 512)
img2_tensor = torch.from_numpy(img2).permute(2,0,1).unsqueeze_(0) 
# img2_tensor shape = (1, 3, 512, 512)

combined_tensor = torch.cat([img1_tensor, img2_tensor], dim=0)
# combined_tensor shape = (2, 3, 512, 512)
predicts = model(combined_tensor)
# predicts shape = (2, 100, 6)
```











