"""
Viewing Stanford 3D Scanning Repository bunny model
"""
# Copyright (c) 2014-2020, Enthought, Inc.
# Standard library imports
import os
from os.path import join
import shutil
import tarfile

# Enthought library imports
from mayavi import mlab

### Download the bunny data, if not already on disk ############################
if not os.path.exists('bunny.tar.gz'):
    # Download the data
    try:
        from urllib import urlopen
    except ImportError:
        from urllib.request import urlopen
    print("Downloading bunny model, Please Wait (3MB)")
    opener = urlopen(
        'http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz')
    with open('bunny.tar.gz', 'wb') as f:
        f.write(opener.read())

# Extract the data
with tarfile.open('bunny.tar.gz') as bunny_tar_file:
    os.makedirs('bunny_data', exist_ok=True)
    bunny_tar_file.extractall('bunny_data')

# Path to the bunny ply file
bunny_ply_file = join('bunny_data', 'bunny', 'reconstruction', 'bun_zipper.ply')

# Optional: copy and rename to bunny.ply in current directory
shutil.copyfile(bunny_ply_file, 'bunny.ply')
print("PLY file saved as bunny.ply")

# Render the bunny ply file
mlab.pipeline.surface(mlab.pipeline.open(bunny_ply_file))
mlab.show()

# Clean up extracted data
shutil.rmtree('bunny_data')
